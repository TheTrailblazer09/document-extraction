"""
CNN baseline training script for document information extraction.

Trains the OCR + CNN baseline on CORD, FUNSD, or SROIE.
OCR results are cached to data/ on first run; subsequent runs skip straight
to training.

Usage:
    python train_baseline_cnn.py --dataset cord
    python train_baseline_cnn.py --dataset cord --no-wandb
    python train_baseline_cnn.py --dataset cord --eval-only --resume cord_cnn_checkpoint
    python train_baseline_cnn.py --dataset funsd --epochs 30 --batch-size 4
"""

import argparse
import json
import os
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent 

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler
from torch.amp import autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report

from src.data.data_loading import load_dataset
from src.data.ocr_processing import OCRDataset, process_dataset, build_grid, NUM_LABELS
from src.models.baseline_cnn import BaselineCNNWrapper

# CLI overrides defaults 

DATASET    = "cord"
DATA_DIR   = str(PROJECT_ROOT / "data")
CACHE_DIR  = str(PROJECT_ROOT / "data" / "cache")
MODELS_DIR = str(PROJECT_ROOT / "models")
GRID_SIZE  = (64, 64)
BATCH_SIZE = 8
NUM_EPOCHS = 30
LR         = 1e-3
WEIGHT_DECAY = 1e-4
WARMUP_STEPS = 100
GRAD_ACCUM   = 1
FP16         = True

# CLI ARGUMENTS!

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset",    type=str,   default=DATASET,  choices=["cord","funsd","sroie"])
    p.add_argument("--data-dir",   type=str,   default=DATA_DIR)
    p.add_argument("--cache-dir",  type=str,   default=CACHE_DIR)
    p.add_argument("--epochs",     type=int,   default=NUM_EPOCHS)
    p.add_argument("--batch-size", type=int,   default=BATCH_SIZE)
    p.add_argument("--lr",         type=float, default=LR)
    p.add_argument("--grad-accum", type=int,   default=GRAD_ACCUM)
    p.add_argument("--grid-size",  type=int,   nargs=2, default=list(GRID_SIZE))
    p.add_argument("--resume",     type=str,   default=None,  help="checkpoint dir to resume from")
    p.add_argument("--eval-only",  action="store_true")
    p.add_argument("--no-wandb",   action="store_true")
    p.add_argument("--num-workers",type=int,   default=0)
    return p.parse_args()

# cache loading helpers 

def cache_path(cache_dir, dataset, split):
    return os.path.join(cache_dir, f"{dataset}_{split}.json")

def build_vocab(items):
    return sorted({w.lower() for item in items for w in item["words"]})

def ensure_cache(dataset_name, data_dir, cache_dir):
    """Run OCR + label assignment once per split; skip if cache exists."""
    required = ["train", "test"] + (["val"] if dataset_name == "cord" else [])
    missing  = [s for s in required if not os.path.exists(cache_path(cache_dir, dataset_name, s))]

    if missing:
        os.makedirs(cache_dir, exist_ok=True)
        print(f"Cache missing for {missing}, loading raw {dataset_name} dataset...")
        ds        = load_dataset(dataset_name, data_dir=data_dir)
        split_map = {"train": ds.train, "val": ds.val, "test": ds.test}
        for split in missing:
            data = split_map.get(split, [])
            if not data:
                continue
            print(f"  Processing {split} → {cache_path(cache_dir, dataset_name, split)}")
            process_dataset(data, cache_path(cache_dir, dataset_name, split))
        del ds  # free image memory before training
    else:
        print("All OCR caches found, skipping dataset load.")

# evaluate model with NER metrics

def run_evaluation(model, eval_data, device, batch_size=8):
    """
    Extract per-word label predictions from the spatial grid and compute
    token-level precision/recall/F1 per class using sklearn.
    """

    model.eval()
    H, W         = model.grid_size
    all_true     = []
    all_pred     = []

    with torch.no_grad():
        for i in range(0, len(eval_data), batch_size):
            batch_items = eval_data[i : i + batch_size]

            grids = torch.stack([
                build_grid(item["words"], item["boxes"], model.embed, H, W)
                for item in batch_items
            ]).to(device)

            pred_grids = model.predict(grids)   # (B, H, W)

            for j, item in enumerate(batch_items):
                pred_grid = pred_grids[j]       # (H, W)
                for (x1, y1, *_), true_label in zip(item["boxes"], item["labels"]):
                    gx = min(int(x1 * W), W - 1)
                    gy = min(int(y1 * H), H - 1)
                    all_pred.append(pred_grid[gy, gx].item())
                    all_true.append(true_label)

    label_ids = list(range(len(model.label_names)))
    report = classification_report(
        all_true, all_pred,
        labels=label_ids,
        target_names=model.label_names,
        output_dict=True,
        zero_division=0,
    )
    model.train()
    return report

# ── Checkpoint ────────────────────────────────────────────────────────────────

def save_checkpoint(model, optimizer, scheduler, scaler, epoch, step, best_f1, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    torch.save({
        "epoch":     epoch,
        "step":      step,
        "best_f1":   best_f1,
        "backbone":  model.backbone.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler":    scaler.state_dict() if scaler else None,
        "vocab":     list(model.embed.vocab.keys()),
        "dataset":   model.dataset_name,
        "grid_size": list(model.grid_size),
    }, os.path.join(save_dir, "checkpoint.pt"))
    print(f"  Checkpoint saved → {save_dir}")

def load_checkpoint(save_dir, model, optimizer=None, scheduler=None, scaler=None):
    ckpt = torch.load(os.path.join(save_dir, "checkpoint.pt"), map_location="cpu", weights_only=False)
    model.backbone.load_state_dict(ckpt["backbone"])
    if optimizer and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler and "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])
    if scaler and ckpt.get("scaler"):
        scaler.load_state_dict(ckpt["scaler"])
    print(f"  Resumed from epoch {ckpt['epoch']}, step {ckpt['step']}, best F1 {ckpt['best_f1']:.4f}")
    return ckpt

# train

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("WARNING: No GPU. Training will be slow.")

    H, W = args.grid_size

    # cached OCR datasets
    ensure_cache(args.dataset, args.data_dir, args.cache_dir)

    train_data = json.load(open(cache_path(args.cache_dir, args.dataset, "train")))
    val_cache  = cache_path(args.cache_dir, args.dataset, "val")
    val_data   = json.load(open(val_cache)) if os.path.exists(val_cache) else []
    test_data  = json.load(open(cache_path(args.cache_dir, args.dataset, "test")))

    vocab = build_vocab(train_data)
    print(f"Vocab size: {len(vocab)}")

    # base model 
    model = BaselineCNNWrapper(
        dataset_name=args.dataset,
        vocab=vocab,
        grid_size=(H, W),
    ).to(device)

    if args.resume:
        ckpt = load_checkpoint(args.resume, model)

    if args.eval_only:
        eval_data = val_data if val_data else test_data
        metrics   = run_evaluation(model, eval_data, device, args.batch_size)
        print_metrics(metrics)
        return

    # load OCR dataset and images 
    train_ds     = OCRDataset(train_data, model.embed, H=H, W=W)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=(device.type=="cuda"))

    print(f"Train: {len(train_data)} items | Val: {len(val_data)} | Test: {len(test_data)}")

    # Optimizer and learning rate schedule
    optimizer    = AdamW(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
    total_steps  = len(train_loader) * args.epochs // args.grad_accum
    warmup_steps = min(WARMUP_STEPS, total_steps // 10)

    scheduler = SequentialLR(
        optimizer,
        schedulers=[
            LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps),
            CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps, eta_min=1e-7),
        ],
        milestones=[warmup_steps],
    )

    use_fp16 = FP16 and device.type == "cuda" # accelerate the training by using half precision.
    scaler   = GradScaler(enabled=use_fp16)

    start_epoch = 0
    global_step = 0
    best_f1     = 0.0

    if args.resume:
        start_epoch = ckpt["epoch"] + 1
        global_step = ckpt["step"]
        best_f1     = ckpt["best_f1"]
        load_checkpoint(args.resume, model, optimizer, scheduler, scaler)

    # wandb integration 
    wandb_run = None
    if not args.no_wandb:
        try:
            import wandb
            wandb_run = wandb.init(
                project="doc-extraction",
                name=f"cnn_{args.dataset}",
                config=vars(args),
                resume="allow" if args.resume else None,
            )
            print(f"W&B: {wandb_run.url}")
        except Exception as e:
            print(f"W&B init failed ({e}), continuing without it.")

    print(f"\n{'='*55}")
    print(f"  Dataset:    {args.dataset}  |  Classes: {model.num_classes}")
    print(f"  Epochs:     {args.epochs}   |  Batch: {args.batch_size}  |  Accum: {args.grad_accum}")
    print(f"  LR:         {args.lr}       |  FP16: {use_fp16}")
    print(f"  Grid:       {H}×{W}         |  Vocab: {len(vocab)}")
    print(f"{'='*55}\n")

    # training loop
    model.train()
    for epoch in range(start_epoch, args.epochs):
        epoch_loss = 0.0
        t0 = time.time()
        optimizer.zero_grad()

        for batch_idx, (grids, label_grids) in enumerate(train_loader):
            grids       = grids.to(device)
            label_grids = label_grids.to(device)

            with autocast(device_type="cuda", enabled=use_fp16):
                loss = model.training_step(grids, label_grids) / args.grad_accum

            scaler.scale(loss).backward()

            if (batch_idx + 1) % args.grad_accum == 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1

            epoch_loss += loss.item() * args.grad_accum

            if wandb_run and global_step % 50 == 0:
                import wandb
                wandb.log({"train/loss": loss.item() * args.grad_accum,
                           "train/lr": optimizer.param_groups[0]["lr"],
                           "train/step": global_step}, step=global_step)

        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{args.epochs}  loss={avg_loss:.4f}  "
              f"lr={optimizer.param_groups[0]['lr']:.2e}  t={time.time()-t0:.0f}s", end="")

        # Evaluate every 5 epochs and on the last epoch
        if (epoch + 1) % 5 == 0 or (epoch + 1) == args.epochs:
            eval_data = val_data if val_data else test_data
            metrics   = run_evaluation(model, eval_data, device, args.batch_size)
            f1        = metrics["macro avg"]["f1-score"]
            print(f"  |  F1={f1:.4f}  P={metrics['macro avg']['precision']:.4f}  "
                  f"R={metrics['macro avg']['recall']:.4f}", end="")

            if wandb_run:
                import wandb
                wandb.log({"eval/macro_f1": f1,
                           "eval/precision": metrics["macro avg"]["precision"],
                           "eval/recall": metrics["macro avg"]["recall"],
                           "eval/epoch": epoch + 1}, step=global_step)

            if f1 > best_f1:
                best_f1  = f1
                best_dir = os.path.join(MODELS_DIR, f"{args.dataset}_cnn_best")
                save_checkpoint(model, optimizer, scheduler, scaler,
                                epoch, global_step, best_f1, best_dir)
                model.save_pretrained(best_dir)
                print(f"  ← new best", end="")

        print()

    # save stuff
    final_dir = os.path.join(MODELS_DIR, f"{args.dataset}_cnn_final")
    save_checkpoint(model, optimizer, scheduler, scaler,
                    args.epochs - 1, global_step, best_f1, final_dir)
    model.save_pretrained(final_dir)

    print(f"\nDone. Best F1: {best_f1:.4f}")
    print(f"Best model  → {best_dir}")
    print(f"Final model → {final_dir}")

    if wandb_run:
        wandb_run.finish()

# helpers

def print_metrics(report: dict):
    print(f"\n  Macro  F1={report['macro avg']['f1-score']:.4f}  "
          f"P={report['macro avg']['precision']:.4f}  "
          f"R={report['macro avg']['recall']:.4f}")
    print(f"  Weighted F1={report['weighted avg']['f1-score']:.4f}\n")
    print(f"  {'Class':<30} {'P':>6} {'R':>6} {'F1':>6} {'N':>6}")
    print(f"  {'-'*54}")
    for cls, m in report.items():
        if isinstance(m, dict) and cls not in ("macro avg", "weighted avg", "accuracy"):
            print(f"  {cls:<30} {m['precision']:>6.3f} {m['recall']:>6.3f} "
                  f"{m['f1-score']:>6.3f} {int(m['support']):>6}")


if __name__ == "__main__":
    train(parse_args())
