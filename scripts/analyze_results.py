"""
Aggregate normalized result JSONs into summary tables for the report.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results"
DEFAULT_ANALYSIS_DIR = DEFAULT_RESULTS_DIR / "analysis"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Aggregate normalized result JSONs into report tables."
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=str(DEFAULT_RESULTS_DIR),
        help="Directory containing normalized result JSON files.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_ANALYSIS_DIR),
        help="Directory for generated tables.",
    )
    return parser.parse_args()


def load_records(results_dir: Path) -> list[dict]:
    records = []
    for path in sorted(results_dir.glob("*.json")):
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if {"model", "dataset", "metrics", "noise_setting"} <= payload.keys():
            payload["_source_file"] = path.name
            records.append(payload)
    return records


def metric_value(metrics: dict, *keys: str):
    for key in keys:
        if key in metrics:
            return metrics[key]
    return None


def build_flat_frame(records: list[dict]) -> pd.DataFrame:
    rows = []
    for record in records:
        metrics = record.get("metrics", {})
        rows.append(
            {
                "model": record.get("model"),
                "dataset": record.get("dataset"),
                "split": record.get("split"),
                "noise_setting": record.get("noise_setting"),
                "run_name": record.get("run_name"),
                "checkpoint": record.get("checkpoint"),
                "primary_score": metric_value(
                    metrics, "field_f1", "f1", "exact_match"
                ),
                "field_f1": metrics.get("field_f1"),
                "ted_accuracy": metrics.get("ted_accuracy"),
                "precision": metric_value(
                    metrics, "field_precision", "precision"
                ),
                "recall": metric_value(
                    metrics, "field_recall", "recall"
                ),
                "exact_match": metrics.get("exact_match"),
                "source_file": record.get("_source_file"),
            }
        )
    return pd.DataFrame(rows)


def dataframe_to_markdown(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = []
    for _, row in frame.iterrows():
        values = [str(row[column]) for column in columns]
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join([header, separator, *rows])


def write_outputs(frame: pd.DataFrame, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    full_path = output_dir / "all_results.csv"
    frame.to_csv(full_path, index=False)

    clean_frame = frame[frame["noise_setting"] == "clean"].copy()
    clean_frame = clean_frame.sort_values(
        by=["dataset", "primary_score"], ascending=[True, False]
    )
    clean_csv = output_dir / "clean_results_summary.csv"
    clean_md = output_dir / "clean_results_summary.md"
    clean_frame.to_csv(clean_csv, index=False)
    clean_md.write_text(dataframe_to_markdown(clean_frame) + "\n", encoding="utf-8")

    robustness = (
        frame.pivot_table(
            index=["model", "dataset"],
            columns="noise_setting",
            values="primary_score",
            aggfunc="max",
        )
        .reset_index()
        .sort_values(by=["dataset", "model"])
    )
    robustness_csv = output_dir / "noise_robustness_summary.csv"
    robustness_md = output_dir / "noise_robustness_summary.md"
    robustness.to_csv(robustness_csv, index=False)
    robustness_md.write_text(
        dataframe_to_markdown(robustness) + "\n", encoding="utf-8"
    )

    print(f"Wrote {full_path}")
    print(f"Wrote {clean_csv}")
    print(f"Wrote {clean_md}")
    print(f"Wrote {robustness_csv}")
    print(f"Wrote {robustness_md}")


def main():
    args = parse_args()
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    records = load_records(results_dir)
    if not records:
        raise SystemExit(
            f"No normalized result JSONs found in {results_dir}. "
            "Run src.evaluation.evaluate first."
        )
    frame = build_flat_frame(records)
    write_outputs(frame, output_dir)


if __name__ == "__main__":
    main()
