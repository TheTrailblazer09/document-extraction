"""
Shared result-schema helpers for experiment outputs.
"""

from __future__ import annotations

import json
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "results"


def build_result_record(
    *,
    model: str,
    dataset: str,
    split: str,
    checkpoint: str,
    run_name: str,
    metrics: dict[str, Any],
    per_field: dict[str, Any] | None,
    config: dict[str, Any],
    noise_setting: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Build one normalized experiment result record.
    """
    record = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "dataset": dataset,
        "split": split,
        "checkpoint": checkpoint,
        "run_name": run_name,
        "metrics": deepcopy(metrics),
        "per_field": deepcopy(per_field or {}),
        "config": deepcopy(config),
        "noise_setting": noise_setting,
    }
    if extra:
        record.update(deepcopy(extra))
    return record


def save_result_record(record: dict[str, Any], output_path: str | Path | None = None) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if output_path is None:
        filename = (
            f"{record['model']}_{record['dataset']}_{record['split']}_"
            f"{record['noise_setting']}_{record['run_name']}.json"
        )
        output_path = RESULTS_DIR / filename
    else:
        output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2)
        handle.write("\n")
    return output_path
