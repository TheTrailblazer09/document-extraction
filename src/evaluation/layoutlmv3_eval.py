"""
seqeval-based evaluation for LayoutLMv3 token classification.

We strip out the special-token positions (labels == -100), map predicted
ids back to BIO tag strings, and hand the resulting sequences to seqeval,
which computes entity-level precision / recall / F1. Entity-level scoring
is the standard for FUNSD and is what the LayoutLMv3 paper reports.

Usage:
    from src.evaluation.layoutlmv3_eval import evaluate_layoutlmv3

    metrics = evaluate_layoutlmv3(
        all_pred_ids, all_label_ids, id2label
    )
"""

from typing import Sequence

import torch


def _ids_to_tag_sequences(
        pred_ids: Sequence[Sequence[int]],
        label_ids: Sequence[Sequence[int]],
        id2label: dict[int, str],
) -> tuple[list[list[str]], list[list[str]]]:
    """
    Drop the -100 positions and convert the rest to BIO tag strings.
    Predictions and labels stay aligned because we filter on the same mask.
    """
    pred_tags: list[list[str]] = []
    true_tags: list[list[str]] = []

    for pred_seq, label_seq in zip(pred_ids, label_ids):
        cur_pred: list[str] = []
        cur_true: list[str] = []
        for p, l in zip(pred_seq, label_seq):
            if l == -100:
                continue
            cur_pred.append(id2label[int(p)])
            cur_true.append(id2label[int(l)])
        pred_tags.append(cur_pred)
        true_tags.append(cur_true)

    return pred_tags, true_tags


def evaluate_layoutlmv3(
        all_pred_ids: Sequence[Sequence[int]],
        all_label_ids: Sequence[Sequence[int]],
        id2label: dict[int, str],
) -> dict:
    """
    Compute entity-level metrics with seqeval.

    Args:
        all_pred_ids:  per-example sequence of predicted label ids
        all_label_ids: per-example sequence of true label ids (with -100
                       at positions to ignore)
        id2label:      mapping from label id to BIO tag string

    Returns:
        dict with overall precision/recall/f1/accuracy and a per-entity
        breakdown.
    """
    # Lazy import so the module is still importable on machines that
    # haven't installed seqeval yet.
    from seqeval.metrics import (
        accuracy_score,
        classification_report,
        f1_score,
        precision_score,
        recall_score,
    )

    pred_tags, true_tags = _ids_to_tag_sequences(
        all_pred_ids, all_label_ids, id2label
    )

    metrics = {
        "precision": precision_score(true_tags, pred_tags, zero_division=0),
        "recall": recall_score(true_tags, pred_tags, zero_division=0),
        "f1": f1_score(true_tags, pred_tags, zero_division=0),
        "accuracy": accuracy_score(true_tags, pred_tags),
        "n_samples": len(true_tags),
    }

    # Per-entity table from seqeval.
    report = classification_report(
        true_tags, pred_tags, output_dict=True, zero_division=0
    )
    per_entity = {}
    for key, vals in report.items():
        if key in ("micro avg", "macro avg", "weighted avg"):
            continue
        if not isinstance(vals, dict):
            continue
        per_entity[key] = {
            "precision": vals.get("precision", 0.0),
            "recall": vals.get("recall", 0.0),
            "f1": vals.get("f1-score", 0.0),
            "support": vals.get("support", 0),
        }
    metrics["per_entity"] = per_entity

    return metrics


def collect_predictions(
        logits: torch.Tensor, labels: torch.Tensor
) -> tuple[list[list[int]], list[list[int]]]:
    """
    Convenience helper for the training loop: turn a batch of logits +
    labels into Python lists in the form evaluate_layoutlmv3 expects.
    """
    preds = logits.argmax(dim=-1).detach().cpu().tolist()
    gold = labels.detach().cpu().tolist()
    return preds, gold
