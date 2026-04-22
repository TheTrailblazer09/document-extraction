"""
Donut fine-tuning script for document information extraction.

Trains Donut (OCR-free document understanding) on CORD, FUNSD, or SROIE.
Designed for local GPU training (16GB+ VRAM recommended).

Usage:
    python -m src.training.train_donut --config configs/donut.yaml --dataset cord
    python -m src.training.train_donut --config configs/donut.yaml --dataset cord --no-wandb --batch-size 1 --num-workers 0
    python -m src.training.train_donut --config configs/donut.yaml --dataset cord --eval-only --resume models/donut_cord_best.pt
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
import yaml

from src.models.donut import DonutModelWrapper, TASK_PROMPTS
from src.data.donut_dataset import DonutDataset
from src.evaluation.donut_eval import evaluate_donut


PROJECT_ROOT = Path(__file__).parent.parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune Donut for document extraction"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/donut.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["cord", "funsd", "sroie"],
        default=None,
        help="Dataset to train on (overrides config)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of epochs (overrides config)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size (overrides config)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Learning rate (overrides config)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable Weights and Biases logging",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Run evaluation only (requires --resume)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="DataLoader workers",
    )
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def apply_cli_overrides(config: dict, args) -> dict:
    if args.dataset is not None:
        config["data"]["dataset"] = args.dataset
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    if args.batch_size is not None:
        config["training"]["batch_size"] = args.batch_size
    if args.lr is not None:
        config["training"]["learning_rate"] = args.lr
    return config


def build_dataloaders(config, processor, task_prompt, num_workers=4):
    dataset_name = config["data"]["dataset"]
    aug_config = config.get("augmentation", None)

    train_dataset = DonutDataset(
        dataset_name=dataset_name,
        split="train",
        processor=processor,
        max_length=config["model"]["max_length"],
        task_prompt=task_prompt,
        augmentation_config=aug_config,
    )

    val_dataset = DonutDataset(
        dataset_name=dataset_name,
        split="val",
        processor=processor,
        max_length=config["model"]["max_length"],
        task_prompt=task_prompt,
        augmentation_config=None,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader


def run_evaluation(model, val_loader, dataset_name, device, max_samples=0):
    model.eval()
    predictions = []
    ground_truths = []
    total_samples = 0

    print("  Running inference on validation set...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"]

            pred_jsons = model.generate(pixel_values, num_beams=1)
            predictions.extend(pred_jsons)

            for label_seq in labels:
                label_seq = label_seq.clone()
                label_seq[label_seq == -100] = model.processor.tokenizer.pad_token_id
                gt_text = model.processor.tokenizer.decode(
                    label_seq, skip_special_tokens=False
                )
                gt_json = model.token_sequence_to_json(gt_text)
                ground_truths.append(gt_json)

            total_samples += pixel_values.size(0)
            if max_samples > 0 and total_samples >= max_samples:
                break

    metrics = evaluate_donut(predictions, ground_truths, dataset_name)
    model.train()
    return metrics


def save_checkpoint(model, optimizer, scheduler, scaler, epoch, global_step, best_f1, config, save_path):
    checkpoint = {
        "epoch": epoch,
        "global_step": global_step,
        "best_f1": best_f1,
        "model_state_dict": model.model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict() if scaler else None,
        "config": config,
    }
    torch.save(checkpoint, save_path)
    print(f"  Checkpoint saved: {save_path}")


def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None, scaler=None):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    if scaler and checkpoint.get("scaler_state_dict"):
        scaler.load_state_dict(checkpoint["scaler_state_dict"])

    print(
        f"  Resumed from epoch {checkpoint['epoch']}, "
        f"step {checkpoint['global_step']}, "
        f"best F1 {checkpoint['best_f1']:.4f}"
    )
    return checkpoint


def train(config, args):
    dataset_name = config["data"]["dataset"]
    task_prompt = TASK_PROMPTS[dataset_name]

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    run_name = (
            config["logging"].get("wandb_run_name")
            or f"donut_{dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")
    else:
        print("WARNING: No GPU detected. Training will be very slow.")

    print(f"\nLoading Donut model: {config['model']['name']}")
    image_size = tuple(config["data"]["image_size"])
    model = DonutModelWrapper(
        pretrained_name=config["model"]["name"],
        max_length=config["model"]["max_length"],
        image_size=image_size,
        dataset_name=dataset_name,
    )
    model = model.to(device)

    print(f"Loading {dataset_name} dataset...")
    train_loader, val_loader = build_dataloaders(
        config, model.processor, task_prompt, num_workers=args.num_workers
    )
    print(f"  Train: {len(train_loader.dataset)} samples")
    print(f"  Val:   {len(val_loader.dataset)} samples")

    if args.eval_only:
        if not args.resume:
            print("ERROR: --eval-only requires --resume checkpoint")
            sys.exit(1)
        load_checkpoint(args.resume, model)
        model = model.to(device)
        metrics = run_evaluation(model, val_loader, dataset_name, device)
        print("\nEvaluation Results:")
        print(f"  TED Accuracy: {metrics['ted_accuracy']:.4f}")
        print(f"  Field F1:     {metrics['field_f1']:.4f}")
        print(f"  Precision:    {metrics['field_precision']:.4f}")
        print(f"  Recall:       {metrics['field_recall']:.4f}")
        print(f"  Exact Match:  {metrics['exact_match']:.4f}")
        if metrics.get("per_field"):
            print("\n  Per-field breakdown:")
            for field, m in metrics["per_field"].items():
                print(f"    {field}: F1={m['f1']:.4f} P={m['precision']:.4f} R={m['recall']:.4f}")
        return

    training_cfg = config["training"]
    optimizer = AdamW(
        model.parameters(),
        lr=training_cfg["learning_rate"],
        weight_decay=training_cfg["weight_decay"],
    )

    total_steps = len(train_loader) * training_cfg["epochs"]
    warmup_steps = training_cfg.get("warmup_steps", 300)

    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.01,
        end_factor=1.0,
        total_iters=warmup_steps,
    )
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=total_steps - warmup_steps,
        eta_min=1e-7,
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_steps],
    )

    use_fp16 = training_cfg.get("fp16", True) and device.type == "cuda"
    scaler = GradScaler("cuda", enabled=use_fp16)

    start_epoch = 0
    global_step = 0
    best_f1 = 0.0
    if args.resume:
        ckpt = load_checkpoint(args.resume, model, optimizer, scheduler, scaler)
        start_epoch = ckpt["epoch"] + 1
        global_step = ckpt["global_step"]
        best_f1 = ckpt["best_f1"]

    wandb_run = None
    if not args.no_wandb:
        try:
            import wandb
            wandb_run = wandb.init(
                project=config["logging"]["wandb_project"],
                name=run_name,
                config=config,
                resume="allow" if args.resume else None,
            )
            print(f"W&B run: {wandb_run.url}")
        except Exception as e:
            print(f"W&B init failed: {e}. Continuing without W&B.")
            wandb_run = None

    grad_accum_steps = training_cfg.get("gradient_accumulation_steps", 1)
    max_grad_norm = training_cfg.get("max_grad_norm", 1.0)
    eval_every = config["evaluation"].get("eval_every_n_epochs", 5)
    epochs = training_cfg["epochs"]

    print(f"\n{'='*60}")
    print(f"Starting training: {run_name}")
    print(f"  Epochs:          {epochs}")
    print(f"  Batch size:      {training_cfg['batch_size']}")
    print(f"  Grad accum:      {grad_accum_steps}")
    print(f"  Effective batch: {training_cfg['batch_size'] * grad_accum_steps}")
    print(f"  Learning rate:   {training_cfg['learning_rate']}")
    print(f"  FP16:            {use_fp16}")
    print(f"  Eval every:      {eval_every} epochs")
    print(f"  Image size:      {image_size}")
    print(f"{'='*60}\n")

    model.train()
    for epoch in range(start_epoch, epochs):
        epoch_loss = 0.0
        epoch_start = time.time()
        optimizer.zero_grad()

        for batch_idx, batch in enumerate(train_loader):
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            with autocast(device_type="cuda", enabled=use_fp16):
                loss = model.training_step(pixel_values, labels)
                loss = loss / grad_accum_steps

            scaler.scale(loss).backward()

            if (batch_idx + 1) % grad_accum_steps == 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1

            epoch_loss += loss.item() * grad_accum_steps

            if global_step % 50 == 0 and global_step > 0:
                lr = optimizer.param_groups[0]["lr"]
                if wandb_run:
                    import wandb
                    wandb.log(
                        {
                            "train/loss": loss.item() * grad_accum_steps,
                            "train/lr": lr,
                            "train/step": global_step,
                        },
                        step=global_step,
                    )

        avg_loss = epoch_loss / len(train_loader)
        elapsed = time.time() - epoch_start
        lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Loss: {avg_loss:.4f} | "
            f"LR: {lr:.2e} | "
            f"Time: {elapsed:.0f}s"
        )

        if (epoch + 1) % eval_every == 0 or (epoch + 1) == epochs:
            print(f"\n  Evaluating at epoch {epoch+1}...")
            eval_start = time.time()
            metrics = run_evaluation(model, val_loader, dataset_name, device)
            eval_time = time.time() - eval_start

            print(
                f"  TED Acc: {metrics['ted_accuracy']:.4f} | "
                f"F1: {metrics['field_f1']:.4f} | "
                f"P: {metrics['field_precision']:.4f} | "
                f"R: {metrics['field_recall']:.4f} | "
                f"EM: {metrics['exact_match']:.4f} | "
                f"Time: {eval_time:.0f}s"
            )

            if wandb_run:
                import wandb
                wandb.log(
                    {
                        "eval/ted_accuracy": metrics["ted_accuracy"],
                        "eval/field_f1": metrics["field_f1"],
                        "eval/field_precision": metrics["field_precision"],
                        "eval/field_recall": metrics["field_recall"],
                        "eval/exact_match": metrics["exact_match"],
                        "eval/epoch": epoch + 1,
                    },
                    step=global_step,
                )

            if metrics["field_f1"] > best_f1:
                best_f1 = metrics["field_f1"]
                best_path = MODELS_DIR / f"donut_{dataset_name}_best.pt"
                save_checkpoint(
                    model, optimizer, scheduler, scaler,
                    epoch, global_step, best_f1, config,
                    str(best_path),
                )
                hf_path = MODELS_DIR / f"donut_{dataset_name}_best_hf"
                model.save_pretrained(str(hf_path))
                print(f"  New best F1: {best_f1:.4f}")

            print()

        if (epoch + 1) % 10 == 0:
            ckpt_path = MODELS_DIR / f"donut_{dataset_name}_epoch_{epoch+1}.pt"
            save_checkpoint(
                model, optimizer, scheduler, scaler,
                epoch, global_step, best_f1, config,
                str(ckpt_path),
            )

    final_path = MODELS_DIR / f"donut_{dataset_name}_final.pt"
    save_checkpoint(
        model, optimizer, scheduler, scaler,
        epochs - 1, global_step, best_f1, config,
        str(final_path),
        )
    model.save_pretrained(str(MODELS_DIR / f"donut_{dataset_name}_final_hf"))

    summary = {
        "run_name": run_name,
        "dataset": dataset_name,
        "best_f1": best_f1,
        "final_epoch": epochs,
        "config": config,
    }
    summary_path = RESULTS_DIR / f"donut_{dataset_name}_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"  Best F1:        {best_f1:.4f}")
    print(f"  Best model:     {MODELS_DIR / f'donut_{dataset_name}_best.pt'}")
    print(f"  Final model:    {final_path}")
    print(f"  Results:        {summary_path}")
    print(f"{'='*60}")

    if wandb_run:
        wandb_run.finish()


def main():
    args = parse_args()
    config = load_config(args.config)
    config = apply_cli_overrides(config, args)
    train(config, args)


if __name__ == "__main__":
    main()