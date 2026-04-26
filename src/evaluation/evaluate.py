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
from torch.utils.data import DataLoader
import yaml

from src.data.donut_dataset import DonutDataset
from src.evaluation.donut_eval import evaluate_donut
from src.evaluation.noise import apply_noise, get_noise_config
from src.evaluation.result_schema import build_result_record, save_result_record
from src.models.donut import DonutModelWrapper


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run shared evaluation and write normalized results."
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["donut", "layoutlmv3", "baseline_cnn"],
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
    return mapping[model_name]


def load_config(config_path: str | Path | None, model_name: str) -> dict[str, Any]:
    config_path = Path(config_path) if config_path else default_config_path(model_name)
    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config.setdefault("evaluation", {})
    config["evaluation"]["shared_config_path"] = str(config_path)
    return config


def checkpoint_load(checkpoint_path: str) -> dict[str, Any]:
    return torch.load(checkpoint_path, map_location="cpu", weights_only=False)


def make_run_name(explicit_name: str | None, model: str, dataset: str, noise_setting: str) -> str:
    if explicit_name:
        return explicit_name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{model}_{dataset}_{noise_setting}_{timestamp}"


def evaluate_donut_adapter(args, config: dict[str, Any]) -> dict[str, Any]:
    checkpoint = checkpoint_load(args.checkpoint)
    dataset_name = args.dataset
    image_size = tuple(config["data"]["image_size"])
    noise_config = get_noise_config(args.noise_setting)

    model = DonutModelWrapper(
        pretrained_name=config["model"]["name"],
        max_length=config["model"]["max_length"],
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
        max_length=config["model"]["max_length"],
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


def evaluate_placeholder_adapter(model_name: str) -> dict[str, Any]:
    raise NotImplementedError(
        f"{model_name} shared evaluation adapter is not implemented yet. "
        "The schema is ready, but this model still needs its adapter."
    )


def main():
    args = parse_args()
    config = load_config(args.config, args.model)
    config["data"]["dataset"] = args.dataset
    config["evaluation"]["noise_setting"] = args.noise_setting

    if args.model == "donut":
        evaluation = evaluate_donut_adapter(args, config)
    elif args.model == "layoutlmv3":
        evaluation = evaluate_placeholder_adapter(args.model)
    else:
        evaluation = evaluate_placeholder_adapter(args.model)

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
