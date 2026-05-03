"""
Shared evaluation entrypoint for document extraction experiments.

Using Bailey's evaluation code and added placeholder adapters for
future LayoutLMv3 and OCR+CNN integration so every model can write the same
result schema once its implementation lands.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import DataLoader, Subset
import yaml

from src.data.data_loading import load_dataset as load_document_dataset
from src.data.donut_dataset import DonutDataset
from src.data.layoutlmv3_dataset import LayoutLMv3Dataset
from src.evaluation.donut_eval import evaluate_donut
from src.evaluation.layoutlmv3_eval import evaluate_layoutlmv3
from src.evaluation.noise import apply_noise, get_noise_config
from src.evaluation.result_schema import build_result_record, save_result_record
from src.models.baseline_cnn import BaselineCNNWrapper
from src.models.donut import DonutModelWrapper
from src.models.layoutlmv3 import LayoutLMv3ModelWrapper
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run shared evaluation and write normalized results."
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["donut", "layoutlmv3", "baseline_cnn", "ocr_cnn", "cnn", "crnn"],
        help="Model adapter to evaluate.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["cord", "funsd", "sroie"],
        help="Dataset split to evaluate.",
    )
    parser.add_argument(
    "--layout-ablation",
    action="store_true",
    help="Evaluate LayoutLMv3 with all bounding boxes zeroed out.",
)
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(PROJECT_ROOT / "data"),
        help="Dataset root for OCR/CNN raw-image evaluation.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Checkpoint path for the selected model.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional YAML config file. Defaults to model-specific config.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        help="Dataset split to evaluate.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Evaluation batch size.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader workers.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Optional cap on evaluated samples. 0 means full split.",
    )
    parser.add_argument(
        "--noise-setting",
        type=str,
        default="clean",
        choices=["clean", "light", "medium", "heavy"],
        help="Named shared noise preset.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional stable run name for the saved result record.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional output JSON path.",
    )
    return parser.parse_args()


def default_config_path(model_name: str) -> Path:
    mapping = {
        "donut": PROJECT_ROOT / "configs" / "donut.yaml",
        "layoutlmv3": PROJECT_ROOT / "configs" / "layoutlmv3.yaml",
        "baseline_cnn": PROJECT_ROOT / "configs" / "baseline_cnn.yaml",
    }
    return mapping[(model_name)]

def default_config(model_name: str, dataset_name: str) -> dict[str, Any]:
    """
    Fallback config used when configs/*.yaml is missing.
    """

    shared = {
        "data": {"dataset": dataset_name},
        "evaluation": {},
    }

    if model_name == "donut":
        shared.update(
            {
                "model": {
                    "name": "naver-clova-ix/donut-base",
                    "max_length": 768,
                },
                "data": {
                    "dataset": dataset_name,
                    "image_size": [1280, 960],
                },
            }
        )
    elif model_name == "layoutlmv3":
        shared.update(
            {
                "model": {
                    "name": "microsoft/layoutlmv3-base",
                    "max_length": 512,
                    "use_2d_position": True,
                }
            }
        )
    else:
        shared.update(
            {
                "model": {
                    "grid_size": [64, 64],
                }
            }
        )

    return shared


def load_config(
    config_path: str | Path | None,
    model_name: str,
    dataset_name: str,
) -> dict[str, Any]:
    if config_path:
        resolved = Path(config_path)
    else:
        resolved = default_config_path(model_name)

    if resolved.exists():
        with open(resolved, "r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle) or {}
    else:
        config = default_config(model_name, dataset_name)

    config.setdefault("data", {})
    config.setdefault("model", {})
    config.setdefault("evaluation", {})
    config["data"]["dataset"] = dataset_name
    config["evaluation"]["shared_config_path"] = str(resolved) if resolved.exists() else None

    return config


def checkpoint_load(checkpoint_path: str) -> dict[str, Any]:
    return torch.load(checkpoint_path, map_location="cpu", weights_only=False)


def make_run_name(explicit_name: str | None, model: str, dataset: str, noise_setting: str) -> str:
    if explicit_name:
        return explicit_name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{model}_{dataset}_{noise_setting}_{timestamp}"

def maybe_subset(dataset, max_samples: int):
    if max_samples and max_samples > 0:
        return Subset(dataset, list(range(min(max_samples, len(dataset)))))
    return dataset


def get_image_from_sample(sample: dict[str, Any]) -> Image.Image:
    img = sample.get("image")

    if isinstance(img, Image.Image):
        return img.convert("RGB")

    if isinstance(img, dict) and "bytes" in img:
        import io

        return Image.open(io.BytesIO(img["bytes"])).convert("RGB")

    raise ValueError(f"Could not extract image from sample. Keys: {list(sample.keys())}")


def select_split(document_dataset, split: str) -> list[dict[str, Any]]:
    split = "val" if split == "validation" else split

    if split == "train":
        return document_dataset.train

    if split == "val":
        return document_dataset.val if document_dataset.val else document_dataset.test

    if split == "test":
        return document_dataset.test

    raise ValueError("split must be train, val/validation, or test")


def evaluate_donut_adapter(args, config: dict[str, Any]) -> dict[str, Any]:
    checkpoint = checkpoint_load(args.checkpoint)
    dataset_name = args.dataset
    image_size = tuple(config["data"].get("image_size", [1280, 960]))
    noise_config = get_noise_config(args.noise_setting)

    model = DonutModelWrapper(
        pretrained_name=config["model"].get("name", "naver-clova-ix/donut-base"),
        max_length=config["model"].get("max_length", 768),
        image_size=image_size,
        dataset_name=dataset_name,
    )
    model.model.load_state_dict(checkpoint["model_state_dict"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    image_transform = None
    if args.noise_setting != "clean":
        image_transform = lambda image: apply_noise(image, noise_config)

    dataset = DonutDataset(
        dataset_name=dataset_name,
        split=args.split,
        processor=model.processor,
        max_length=config["model"].get("max_length", 768),
        task_prompt=model.task_prompt,
        augmentation_config=None,
        image_transform=image_transform,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    predictions: list[str] = []
    ground_truths: list[str] = []
    evaluated_samples = 0

    with torch.no_grad():
        for batch in loader:
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"]
            predictions.extend(model.generate(pixel_values, num_beams=1))

            for label_seq in labels:
                label_seq = label_seq.clone()
                label_seq[label_seq == -100] = model.processor.tokenizer.pad_token_id
                gt_text = model.processor.tokenizer.decode(
                    label_seq, skip_special_tokens=False
                )
                ground_truths.append(model.token_sequence_to_json(gt_text))

            evaluated_samples += pixel_values.size(0)
            if args.max_samples > 0 and evaluated_samples >= args.max_samples:
                predictions = predictions[: args.max_samples]
                ground_truths = ground_truths[: args.max_samples]
                break

    metrics = evaluate_donut(predictions, ground_truths, dataset_name)
    per_field = metrics.pop("per_field", {})

    extra = {
        "adapter": "donut",
        "num_samples": metrics.get("n_samples", len(predictions)),
        "noise_config": noise_config,
        "training_checkpoint_metadata": {
            "epoch": checkpoint.get("epoch"),
            "global_step": checkpoint.get("global_step"),
            "best_f1": checkpoint.get("best_f1"),
        },
    }
    return {
        "metrics": metrics,
        "per_field": per_field,
        "extra": extra,
    }


def load_layoutlmv3_model(args, config: dict[str, Any], device: torch.device):
    dataset_name = args.dataset
    use_2d_position = not args.layout_ablation

    if args.layout_ablation:
        config["model"]["use_2d_position"] = False

    checkpoint_path = Path(args.checkpoint)

    if checkpoint_path.is_dir():
        model = LayoutLMv3ModelWrapper.from_pretrained(
            str(checkpoint_path),
            dataset_name=dataset_name,
            use_2d_position=use_2d_position,
        )
        checkpoint_meta: dict[str, Any] = {}
    else:
        checkpoint = checkpoint_load(checkpoint_path)

        model = LayoutLMv3ModelWrapper(
            pretrained_name=config["model"].get("name", "microsoft/layoutlmv3-base"),
            dataset_name=dataset_name,
            use_2d_position=use_2d_position,
        )

        state = (
            checkpoint.get("model_state_dict")
            or checkpoint.get("state_dict")
            or checkpoint.get("model")
            or checkpoint.get("layoutlmv3")
            or checkpoint
        )

        try:
            model.model.load_state_dict(state)
        except RuntimeError:
            # Some checkpoints may have been saved from the wrapper itself.
            model.load_state_dict(state)

        checkpoint_meta = {
            "epoch": checkpoint.get("epoch") if isinstance(checkpoint, dict) else None,
            "global_step": checkpoint.get("global_step", checkpoint.get("step"))
            if isinstance(checkpoint, dict)
            else None,
            "best_f1": checkpoint.get("best_f1") if isinstance(checkpoint, dict) else None,
        }

    return model.to(device), checkpoint_meta


def evaluate_layoutlmv3_adapter(args, config: dict[str, Any]) -> dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    noise_config = get_noise_config(args.noise_setting)

    model, checkpoint_meta = load_layoutlmv3_model(args, config, device)
    model.eval()

    image_transform = None
    if args.noise_setting != "clean":
        image_transform = lambda image: apply_noise(image, noise_config)

    dataset = LayoutLMv3Dataset(
        dataset_name=args.dataset,
        split=args.split,
        processor=model.processor,
        label2id=model.label2id,
        max_length=config["model"].get("max_length", 512),
        augmentation_config=None,
        image_transform=image_transform,
    )
    dataset = maybe_subset(dataset, args.max_samples)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    all_pred_ids: list[list[int]] = []
    all_label_ids: list[list[int]] = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            bbox = batch["bbox"].to(device)
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"]

            pred_ids = model.predict(
                input_ids=input_ids,
                attention_mask=attention_mask,
                bbox=bbox,
                pixel_values=pixel_values,
            )

            all_pred_ids.extend(pred_ids.detach().cpu().tolist())
            all_label_ids.extend(labels.detach().cpu().tolist())

    metrics = evaluate_layoutlmv3(all_pred_ids, all_label_ids, model.id2label)
    per_field = metrics.pop("per_entity", {})

    metrics["field_precision"] = metrics.get("precision", 0.0)
    metrics["field_recall"] = metrics.get("recall", 0.0)
    metrics["field_f1"] = metrics.get("f1", 0.0)

    extra = {
        "adapter": "layoutlmv3",
        "num_samples": metrics.get("n_samples", len(all_label_ids)),
        "noise_config": noise_config,
        "layout_ablation": args.layout_ablation,
        "training_checkpoint_metadata": checkpoint_meta,
    }

    return {
        "metrics": metrics,
        "per_field": per_field,
        "extra": extra,
    }

def load_baseline_cnn_model(args, device: torch.device):
    checkpoint_path = Path(args.checkpoint)
    checkpoint_meta: dict[str, Any] = {}

    if checkpoint_path.is_dir() and (checkpoint_path / "config.json").exists():
        model = BaselineCNNWrapper.from_pretrained(str(checkpoint_path))

        ckpt_file = checkpoint_path / "checkpoint.pt"
        if ckpt_file.exists():
            ckpt = checkpoint_load(ckpt_file)
            checkpoint_meta = {
                "epoch": ckpt.get("epoch"),
                "global_step": ckpt.get("step", ckpt.get("global_step")),
                "best_f1": ckpt.get("best_f1"),
            }

    else:
        ckpt_file = checkpoint_path / "checkpoint.pt" if checkpoint_path.is_dir() else checkpoint_path
        ckpt = checkpoint_load(ckpt_file)

        dataset_name = ckpt.get("dataset", args.dataset)
        vocab = ckpt.get("vocab")
        grid_size = tuple(ckpt.get("grid_size", [64, 64]))

        if vocab is None:
            raise ValueError(
                "Baseline CNN checkpoint must include 'vocab' or be a "
                "save_pretrained directory containing config.json."
            )

        model = BaselineCNNWrapper(
            dataset_name=dataset_name,
            vocab=vocab,
            grid_size=grid_size,
        )

        state = ckpt.get("backbone", ckpt.get("model_state_dict", ckpt))
        model.backbone.load_state_dict(state)

        checkpoint_meta = {
            "epoch": ckpt.get("epoch"),
            "global_step": ckpt.get("step", ckpt.get("global_step")),
            "best_f1": ckpt.get("best_f1"),
        }

    return model.to(device), checkpoint_meta


def classification_report_to_metrics(
    report: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    macro = report.get("macro avg", {})
    weighted = report.get("weighted avg", {})

    metrics = {
        "precision": macro.get("precision", 0.0),
        "recall": macro.get("recall", 0.0),
        "f1": macro.get("f1-score", 0.0),
        "field_precision": macro.get("precision", 0.0),
        "field_recall": macro.get("recall", 0.0),
        "field_f1": macro.get("f1-score", 0.0),
        "weighted_precision": weighted.get("precision", 0.0),
        "weighted_recall": weighted.get("recall", 0.0),
        "weighted_f1": weighted.get("f1-score", 0.0),
        "accuracy": report.get("accuracy", 0.0),
    }

    per_field = {}
    for key, value in report.items():
        if key in {"accuracy", "macro avg", "weighted avg"}:
            continue
        if not isinstance(value, dict):
            continue

        per_field[key] = {
            "precision": value.get("precision", 0.0),
            "recall": value.get("recall", 0.0),
            "f1": value.get("f1-score", 0.0),
            "support": value.get("support", 0),
        }

    return metrics, per_field


def evaluate_baseline_cnn_adapter(args, config: dict[str, Any]) -> dict[str, Any]:
    from sklearn.metrics import classification_report

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    noise_config = get_noise_config(args.noise_setting)

    model, checkpoint_meta = load_baseline_cnn_model(args, device)
    model.eval()

    raw_dataset = load_document_dataset(args.dataset, data_dir=args.data_dir)
    eval_samples = select_split(raw_dataset, args.split)

    if args.max_samples and args.max_samples > 0:
        eval_samples = eval_samples[: args.max_samples]

    H, W = model.grid_size
    all_true: list[int] = []
    all_pred: list[int] = []
    skipped_empty_ocr = 0

    with torch.no_grad():
        for sample in eval_samples:
            image = get_image_from_sample(sample)

            if args.noise_setting != "clean":
                image = apply_noise(image, noise_config)

            prep = model.prepare_input(image, sample)

            words = prep.get("words", [])
            boxes = prep.get("boxes", [])
            true_labels = prep.get("labels", [])

            if not words or not boxes or not true_labels:
                skipped_empty_ocr += 1
                continue

            grid = prep["grid"].unsqueeze(0).to(device)
            pred_grid = model.predict(grid).squeeze(0)

            for (x1, y1, *_), true_label in zip(boxes, true_labels):
                gx = min(max(int(x1 * W), 0), W - 1)
                gy = min(max(int(y1 * H), 0), H - 1)

                all_pred.append(int(pred_grid[gy, gx].item()))
                all_true.append(int(true_label))

    if not all_true:
        metrics = {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "field_precision": 0.0,
            "field_recall": 0.0,
            "field_f1": 0.0,
            "weighted_precision": 0.0,
            "weighted_recall": 0.0,
            "weighted_f1": 0.0,
            "accuracy": 0.0,
        }
        per_field = {}
    else:
        label_ids = list(range(len(model.label_names)))
        report = classification_report(
            all_true,
            all_pred,
            labels=label_ids,
            target_names=model.label_names,
            output_dict=True,
            zero_division=0,
        )
        metrics, per_field = classification_report_to_metrics(report)

    metrics["n_samples"] = len(eval_samples)
    metrics["n_tokens"] = len(all_true)

    extra = {
        "adapter": "baseline_cnn",
        "num_samples": len(eval_samples),
        "num_tokens": len(all_true),
        "skipped_empty_ocr": skipped_empty_ocr,
        "noise_config": noise_config,
        "training_checkpoint_metadata": checkpoint_meta,
    }

    return {
        "metrics": metrics,
        "per_field": per_field,
        "extra": extra,
    }

def main():
    args = parse_args()

    config = load_config(args.config, args.model, args.dataset)
    config["evaluation"]["noise_setting"] = args.noise_setting

    if args.model == "donut":
        evaluation = evaluate_donut_adapter(args, config)
    elif args.model == "layoutlmv3":
        evaluation = evaluate_layoutlmv3_adapter(args, config)
    elif args.model == "baseline_cnn":
        evaluation = evaluate_baseline_cnn_adapter(args, config)
    else:
        raise ValueError(f"Unknown model: {args.model}")
    run_name = make_run_name(args.run_name, args.model, args.dataset, args.noise_setting)
    record = build_result_record(
        model=args.model,
        dataset=args.dataset,
        split=args.split,
        checkpoint=args.checkpoint,
        run_name=run_name,
        metrics=evaluation["metrics"],
        per_field=evaluation["per_field"],
        config=config,
        noise_setting=args.noise_setting,
        extra=evaluation["extra"],
    )
    output_path = save_result_record(record, args.output)

    print(json.dumps(record["metrics"], indent=2))
    print(f"Saved normalized result record to {output_path}")


if __name__ == "__main__":
    main()
