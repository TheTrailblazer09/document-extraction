"""
LayoutLMv3 model wrapper for token classification on document IE datasets.

LayoutLMv3 (Microsoft, 2022) is a multimodal transformer for document
understanding. It fuses three streams:
  - text   (subword tokens)
  - layout (2D bounding-box position embeddings)
  - vision (image patches via a ViT-style encoder)

This wrapper handles fine-tuning for token classification (BIO tagging),
which is the standard formulation for FUNSD and the natural framing for
CORD when we want to compare apples-to-apples against an OCR + sequence
labeler pipeline.

The `use_2d_position` flag controls whether the model receives real
bounding boxes or a zeroed-out layout signal. Setting it to False
effectively disables the 2D position embeddings -- this is the layout
ablation experiment.

Paper: https://arxiv.org/abs/2204.08387
Pretrained: microsoft/layoutlmv3-base
"""

from typing import Optional

import torch
import torch.nn as nn
from transformers import (
    LayoutLMv3ForTokenClassification,
    LayoutLMv3Processor,
)


# Standard FUNSD BIO tag set (matches nielsr/funsd-layoutlmv3).
FUNSD_LABELS = [
    "O",
    "B-HEADER", "I-HEADER",
    "B-QUESTION", "I-QUESTION",
    "B-ANSWER", "I-ANSWER",
]

# CORD has fine-grained semantic categories on each line. We tag each word
# with B-/I- relative to its line's category. The full set is fixed here
# so that label ids stay stable across runs and checkpoints.
CORD_LABELS = [
    "O",
    "B-MENU.NM", "I-MENU.NM",
    "B-MENU.NUM", "I-MENU.NUM",
    "B-MENU.UNITPRICE", "I-MENU.UNITPRICE",
    "B-MENU.CNT", "I-MENU.CNT",
    "B-MENU.DISCOUNTPRICE", "I-MENU.DISCOUNTPRICE",
    "B-MENU.PRICE", "I-MENU.PRICE",
    "B-MENU.ITEMSUBTOTAL", "I-MENU.ITEMSUBTOTAL",
    "B-MENU.VATYN", "I-MENU.VATYN",
    "B-MENU.ETC", "I-MENU.ETC",
    "B-MENU.SUB.NM", "I-MENU.SUB.NM",
    "B-MENU.SUB.UNITPRICE", "I-MENU.SUB.UNITPRICE",
    "B-MENU.SUB.CNT", "I-MENU.SUB.CNT",
    "B-MENU.SUB.PRICE", "I-MENU.SUB.PRICE",
    "B-MENU.SUB.ETC", "I-MENU.SUB.ETC",
    "B-VOID_MENU.NM", "I-VOID_MENU.NM",
    "B-VOID_MENU.PRICE", "I-VOID_MENU.PRICE",
    "B-SUB_TOTAL.SUBTOTAL_PRICE", "I-SUB_TOTAL.SUBTOTAL_PRICE",
    "B-SUB_TOTAL.DISCOUNT_PRICE", "I-SUB_TOTAL.DISCOUNT_PRICE",
    "B-SUB_TOTAL.SERVICE_PRICE", "I-SUB_TOTAL.SERVICE_PRICE",
    "B-SUB_TOTAL.OTHERSVC_PRICE", "I-SUB_TOTAL.OTHERSVC_PRICE",
    "B-SUB_TOTAL.TAX_PRICE", "I-SUB_TOTAL.TAX_PRICE",
    "B-SUB_TOTAL.ETC", "I-SUB_TOTAL.ETC",
    "B-TOTAL.TOTAL_PRICE", "I-TOTAL.TOTAL_PRICE",
    "B-TOTAL.TOTAL_ETC", "I-TOTAL.TOTAL_ETC",
    "B-TOTAL.CASHPRICE", "I-TOTAL.CASHPRICE",
    "B-TOTAL.CHANGEPRICE", "I-TOTAL.CHANGEPRICE",
    "B-TOTAL.CREDITCARDPRICE", "I-TOTAL.CREDITCARDPRICE",
    "B-TOTAL.EMONEYPRICE", "I-TOTAL.EMONEYPRICE",
    "B-TOTAL.MENUTYPE_CNT", "I-TOTAL.MENUTYPE_CNT",
    "B-TOTAL.MENUQTY_CNT", "I-TOTAL.MENUQTY_CNT",
]


DATASET_LABELS = {
    "funsd": FUNSD_LABELS,
    "cord": CORD_LABELS,
}


def get_label_maps(dataset_name: str) -> tuple[list[str], dict[str, int], dict[int, str]]:
    """Return (labels, label2id, id2label) for the given dataset."""
    if dataset_name not in DATASET_LABELS:
        raise ValueError(
            f"No label set defined for dataset '{dataset_name}'. "
            f"Available: {list(DATASET_LABELS.keys())}"
        )
    labels = DATASET_LABELS[dataset_name]
    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for i, l in enumerate(labels)}
    return labels, label2id, id2label


class LayoutLMv3ModelWrapper(nn.Module):
    """
    Wraps HuggingFace's LayoutLMv3ForTokenClassification.

    Responsibilities:
      - Load the pretrained backbone with a fresh classifier head sized
        to the dataset's label set.
      - Hold the matching processor (image + tokenizer + bbox handling).
      - Optionally zero out bounding boxes so the model can't use 2D
        position information (layout ablation).
    """

    def __init__(
            self,
            pretrained_name: str = "microsoft/layoutlmv3-base",
            dataset_name: str = "cord",
            use_2d_position: bool = True,
            apply_ocr: bool = False,
    ):
        super().__init__()
        self.pretrained_name = pretrained_name
        self.dataset_name = dataset_name
        self.use_2d_position = use_2d_position

        labels, label2id, id2label = get_label_maps(dataset_name)
        self.labels = labels
        self.label2id = label2id
        self.id2label = id2label
        self.num_labels = len(labels)

        # apply_ocr=False because we provide words + boxes ourselves.
        self.processor = LayoutLMv3Processor.from_pretrained(
            pretrained_name, apply_ocr=apply_ocr
        )

        self.model = LayoutLMv3ForTokenClassification.from_pretrained(
            pretrained_name,
            num_labels=self.num_labels,
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True,
        )

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def _maybe_zero_bbox(self, bbox: torch.Tensor) -> torch.Tensor:
        """For the layout ablation, replace all boxes with (0,0,0,0)."""
        if self.use_2d_position:
            return bbox
        return torch.zeros_like(bbox)

    def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            bbox: torch.Tensor,
            pixel_values: torch.Tensor,
            labels: Optional[torch.Tensor] = None,
    ):
        bbox = self._maybe_zero_bbox(bbox)
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            bbox=bbox,
            pixel_values=pixel_values,
            labels=labels,
        )

    def training_step(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            bbox: torch.Tensor,
            pixel_values: torch.Tensor,
            labels: torch.Tensor,
    ) -> torch.Tensor:
        outputs = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            bbox=bbox,
            pixel_values=pixel_values,
            labels=labels,
        )
        return outputs.loss

    @torch.no_grad()
    def predict(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            bbox: torch.Tensor,
            pixel_values: torch.Tensor,
    ) -> torch.Tensor:
        """Return predicted label ids of shape [B, seq_len]."""
        outputs = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            bbox=bbox,
            pixel_values=pixel_values,
        )
        return outputs.logits.argmax(dim=-1)

    def save_pretrained(self, save_dir: str):
        self.model.save_pretrained(save_dir)
        self.processor.save_pretrained(save_dir)

    @classmethod
    def from_pretrained(
            cls,
            save_dir: str,
            dataset_name: str = "cord",
            use_2d_position: bool = True,
    ):
        wrapper = cls.__new__(cls)
        nn.Module.__init__(wrapper)
        wrapper.pretrained_name = save_dir
        wrapper.dataset_name = dataset_name
        wrapper.use_2d_position = use_2d_position

        labels, label2id, id2label = get_label_maps(dataset_name)
        wrapper.labels = labels
        wrapper.label2id = label2id
        wrapper.id2label = id2label
        wrapper.num_labels = len(labels)

        wrapper.processor = LayoutLMv3Processor.from_pretrained(
            save_dir, apply_ocr=False
        )
        wrapper.model = LayoutLMv3ForTokenClassification.from_pretrained(save_dir)
        return wrapper
