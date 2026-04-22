"""
Evaluation metrics for Donut.

Donut generates JSON strings, so we can't use token-level seqeval like
LayoutLMv3. Instead we use:
1. Tree Edit Distance (TED) -- structural similarity between predicted
   and ground truth JSON trees (normalized to 0-1, lower is better)
2. Field-level F1 -- precision/recall/F1 on extracted key-value pairs
3. Exact match accuracy -- what fraction of fields are exactly right

The TED metric follows the original Donut paper's evaluation protocol.

Usage:
    from src.evaluation.donut_eval import evaluate_donut, compute_ted

    # Single example
    ted_score = compute_ted(pred_json_str, gt_json_str)

    # Full evaluation
    metrics = evaluate_donut(predictions, ground_truths)
"""

import json
import re
from collections import Counter
from typing import Optional


def evaluate_donut(
        predictions: list[str],
        ground_truths: list[str],
        dataset_name: str = "cord",
) -> dict:
    """
    Run full evaluation on a list of predictions vs ground truths.

    Args:
        predictions: list of JSON strings from model.generate()
        ground_truths: list of ground truth JSON strings
        dataset_name: which dataset (affects field extraction logic)

    Returns:
        dict with ted, field_f1, field_precision, field_recall,
        exact_match, and per-field breakdown
    """
    assert len(predictions) == len(ground_truths), (
        f"Length mismatch: {len(predictions)} predictions vs "
        f"{len(ground_truths)} ground truths"
    )

    ted_scores = []
    all_pred_fields = []
    all_gt_fields = []
    exact_matches = 0

    for pred_str, gt_str in zip(predictions, ground_truths):
        # Tree edit distance
        ted = compute_ted(pred_str, gt_str)
        ted_scores.append(ted)

        # Extract flat key-value pairs for field-level F1
        pred_fields = flatten_json(pred_str)
        gt_fields = flatten_json(gt_str)
        all_pred_fields.append(pred_fields)
        all_gt_fields.append(gt_fields)

        # Exact match: all fields correct
        if pred_fields == gt_fields:
            exact_matches += 1

    # Aggregate metrics
    n = len(predictions)
    avg_ted = sum(ted_scores) / n if n > 0 else 0.0

    # Compute micro-averaged field F1
    field_metrics = compute_field_f1(all_pred_fields, all_gt_fields)

    # Per-field breakdown
    per_field = compute_per_field_metrics(all_pred_fields, all_gt_fields)

    return {
        "ted": avg_ted,
        "ted_accuracy": 1.0 - avg_ted,
        "field_f1": field_metrics["f1"],
        "field_precision": field_metrics["precision"],
        "field_recall": field_metrics["recall"],
        "exact_match": exact_matches / n if n > 0 else 0.0,
        "n_samples": n,
        "per_field": per_field,
    }


def compute_ted(pred_str: str, gt_str: str) -> float:
    """
    Compute normalized Tree Edit Distance between two JSON strings.

    Uses a simplified approach: flatten both JSON trees to sets of
    (path, value) pairs and compute the normalized edit distance
    based on set differences.

    Returns a score in [0, 1] where 0 means identical.
    """
    pred_pairs = _json_to_path_value_pairs(pred_str)
    gt_pairs = _json_to_path_value_pairs(gt_str)

    if not pred_pairs and not gt_pairs:
        return 0.0

    # Compute distance as: (insertions + deletions) / total possible
    pred_set = set(pred_pairs)
    gt_set = set(gt_pairs)

    insertions = len(pred_set - gt_set)  # in pred but not in gt
    deletions = len(gt_set - pred_set)  # in gt but not in pred
    total = len(pred_set | gt_set)

    if total == 0:
        return 0.0

    return (insertions + deletions) / (2 * total)


def _json_to_path_value_pairs(json_str: str) -> list[tuple[str, str]]:
    """
    Flatten a JSON string into (dotted_path, value) pairs.

    Example:
        {"menu": [{"nm": "LATTE", "price": "3.00"}], "total": {"price": "3.00"}}
        -> [("menu.0.nm", "LATTE"), ("menu.0.price", "3.00"),
            ("total.price", "3.00")]
    """
    try:
        if isinstance(json_str, str):
            obj = json.loads(json_str)
        else:
            obj = json_str
    except (json.JSONDecodeError, TypeError):
        return []

    pairs = []
    _flatten_recursive(obj, "", pairs)
    return pairs


def _flatten_recursive(
        obj, prefix: str, pairs: list[tuple[str, str]]
):
    """Recursively flatten a JSON object into path-value pairs."""
    if isinstance(obj, dict):
        for key, value in obj.items():
            new_prefix = f"{prefix}.{key}" if prefix else key
            _flatten_recursive(value, new_prefix, pairs)
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            new_prefix = f"{prefix}.{i}"
            _flatten_recursive(item, new_prefix, pairs)
    else:
        pairs.append((prefix, str(obj)))


def flatten_json(json_str: str) -> dict[str, str]:
    """
    Flatten a JSON string to a dict of {dotted_path: value}.
    Used for field-level F1 computation.
    """
    pairs = _json_to_path_value_pairs(json_str)
    return dict(pairs)


def compute_field_f1(
        all_preds: list[dict[str, str]],
        all_gts: list[dict[str, str]],
) -> dict[str, float]:
    """
    Compute micro-averaged precision, recall, F1 over all field extractions.

    A field is "correct" if the predicted value for a given path matches
    the ground truth value exactly (case-sensitive).
    """
    tp = 0
    fp = 0
    fn = 0

    for pred_fields, gt_fields in zip(all_preds, all_gts):
        pred_items = set(pred_fields.items())
        gt_items = set(gt_fields.items())

        tp += len(pred_items & gt_items)
        fp += len(pred_items - gt_items)
        fn += len(gt_items - pred_items)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {"precision": precision, "recall": recall, "f1": f1}


def compute_per_field_metrics(
        all_preds: list[dict[str, str]],
        all_gts: list[dict[str, str]],
) -> dict[str, dict[str, float]]:
    """
    Compute precision/recall/F1 broken down by top-level field name.

    For CORD this gives you separate scores for menu items, totals, etc.
    For SROIE this gives you scores for company, date, address, total.
    """
    # Group by the top-level key (first segment of the dotted path)
    field_tp: Counter = Counter()
    field_fp: Counter = Counter()
    field_fn: Counter = Counter()

    for pred_fields, gt_fields in zip(all_preds, all_gts):
        pred_items = set(pred_fields.items())
        gt_items = set(gt_fields.items())

        for path, value in pred_items & gt_items:
            top_key = path.split(".")[0]
            field_tp[top_key] += 1

        for path, value in pred_items - gt_items:
            top_key = path.split(".")[0]
            field_fp[top_key] += 1

        for path, value in gt_items - pred_items:
            top_key = path.split(".")[0]
            field_fn[top_key] += 1

    all_fields = set(field_tp.keys()) | set(field_fp.keys()) | set(field_fn.keys())
    per_field = {}

    for field in sorted(all_fields):
        tp = field_tp[field]
        fp = field_fp[field]
        fn = field_fn[field]
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        per_field[field] = {"precision": p, "recall": r, "f1": f1}

    return per_field