"""
Dataset for LayoutLMv3 token-classification fine-tuning on CORD and FUNSD.

LayoutLMv3 takes (image, words, boxes, word_labels) and emits one label
per subword token. The boxes must be normalized to the 0-1000 range that
LayoutLMv3 was pretrained with.

Two adapters live here:
  - FUNSD (nielsr/funsd-layoutlmv3): ships with tokens / bboxes / ner_tags
    already normalized -- we just rewire the field names and pass through.
  - CORD  (naver-clova-ix/cord-v2):  ships with a JSON ground_truth that
    contains a `valid_line` array of {words: [{text, quad}], category}.
    We flatten that into per-word (text, box, BIO label).

Usage:
    from src.data.layoutlmv3_dataset import LayoutLMv3Dataset

    dataset = LayoutLMv3Dataset(
        dataset_name="cord",
        split="train",
        processor=processor,
        label2id=label2id,
        max_length=512,
    )
"""

import json
from pathlib import Path
from typing import Optional

import torch
from PIL import Image
from torch.utils.data import Dataset

try:
    import albumentations as A
    import numpy as np

    ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    ALBUMENTATIONS_AVAILABLE = False


DATA_DIR = Path(__file__).parent.parent.parent / "data"


class LayoutLMv3Dataset(Dataset):
    """
    Token-classification dataset for LayoutLMv3.

    Returns batches of dicts with:
      - input_ids       [seq_len]
      - attention_mask  [seq_len]
      - bbox            [seq_len, 4]
      - pixel_values    [3, H, W]
      - labels          [seq_len]  (-100 for special / padding tokens)
    """

    def __init__(
        self,
        dataset_name: str,
        split: str,
        processor,
        label2id: dict[str, int],
        max_length: int = 512,
        augmentation_config: Optional[dict] = None,
        image_transform=None,
):
        self.dataset_name = dataset_name
        self.split = split
        self.processor = processor
        self.label2id = label2id
        self.max_length = max_length

        self.data = self._load_dataset(dataset_name, split)
        self.image_transform = image_transform

        self.augment = None
        if augmentation_config and augmentation_config.get("enabled", False):
            if not ALBUMENTATIONS_AVAILABLE:
                print(
                    "WARNING: albumentations not installed, "
                    "skipping augmentation"
                )
            else:
                # Note: we don't rotate here because rotating the image
                # without rotating the bboxes would produce mislabeled
                # supervision. Keep augmentation to pixel-only ops.
                self.augment = A.Compose(
                    [
                        A.GaussianBlur(
                            blur_limit=augmentation_config.get("blur_limit", 5),
                            p=0.3,
                        ),
                        A.ImageCompression(
                            quality_lower=augmentation_config.get(
                                "quality_lower", 50
                            ),
                            quality_upper=100,
                            p=0.3,
                        ),
                    ]
                )

    def _load_dataset(self, dataset_name: str, split: str):
        from datasets import load_from_disk

        dataset_path = DATA_DIR / dataset_name / "hf_dataset"
        if not dataset_path.exists():
            raise FileNotFoundError(
                f"Dataset not found at {dataset_path}. "
                f"Run: python scripts/download_data.py --dataset {dataset_name}"
            )

        full_dataset = load_from_disk(str(dataset_path))

        if split in ("val", "validation"):
            if "validation" in full_dataset:
                return full_dataset["validation"]
            if "test" in full_dataset:
                return full_dataset["test"]
            raise ValueError(f"No validation/test split in {dataset_name}")
        if split in full_dataset:
            return full_dataset[split]
        raise ValueError(
            f"Split '{split}' not found. Available: {list(full_dataset.keys())}"
        )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        sample = self.data[idx]
        image = self._get_image(sample)
        words, boxes, word_labels = self._get_words_boxes_labels(sample, image)
        if self.image_transform is not None:
            image = self.image_transform(image)

        if self.augment is not None:
            image_np = np.array(image)
            image = Image.fromarray(self.augment(image=image_np)["image"])

        # Empty-document guard: the processor needs at least one word.
        if not words:
            words = [""]
            boxes = [[0, 0, 0, 0]]
            word_labels = [self.label2id.get("O", 0)]

        encoded = self.processor(
            image,
            words,
            boxes=boxes,
            word_labels=word_labels,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "bbox": encoded["bbox"].squeeze(0),
            "pixel_values": encoded["pixel_values"].squeeze(0),
            "labels": encoded["labels"].squeeze(0),
        }

    def _get_image(self, sample) -> Image.Image:
        img = sample.get("image")
        if isinstance(img, Image.Image):
            return img.convert("RGB")
        if isinstance(img, dict) and "bytes" in img:
            import io

            return Image.open(io.BytesIO(img["bytes"])).convert("RGB")
        raise ValueError(
            f"Could not extract image from sample. Keys: {list(sample.keys())}"
        )

    def _get_words_boxes_labels(
            self, sample, image: Image.Image
    ) -> tuple[list[str], list[list[int]], list[int]]:
        if self.dataset_name == "funsd":
            return self._funsd_fields(sample)
        if self.dataset_name == "cord":
            return self._cord_fields(sample, image)
        raise ValueError(f"Unsupported dataset: {self.dataset_name}")

    def _funsd_fields(
            self, sample
    ) -> tuple[list[str], list[list[int]], list[int]]:
        """
        nielsr/funsd-layoutlmv3 already provides everything in the right shape:
            tokens   : list[str]
            bboxes   : list[[x0, y0, x1, y1]]  (already normalized to 0-1000)
            ner_tags : list[int]               (matches FUNSD_LABELS order)
        """
        words = list(sample.get("tokens", sample.get("words", [])))
        boxes = [list(b) for b in sample["bboxes"]]
        labels = list(sample["ner_tags"])
        return words, boxes, labels

    def _cord_fields(
            self, sample, image: Image.Image
    ) -> tuple[list[str], list[list[int]], list[int]]:
        """
        Parse CORD's ground_truth JSON into LayoutLMv3 inputs.

        valid_line entries look like:
            {
              "words": [
                {"text": "AMERICANO", "quad": {x1,y1,x2,y2,x3,y3,x4,y4}},
                ...
              ],
              "category": "menu.nm",
              "group_id": 1
            }
        We flatten to one (word, box, label) tuple per word, BIO-tagged
        within each line.
        """
        gt_raw = sample.get("ground_truth", "")
        gt = json.loads(gt_raw) if isinstance(gt_raw, str) else gt_raw
        valid_lines = gt.get("valid_line", [])

        img_w, img_h = image.size
        o_id = self.label2id["O"]

        words: list[str] = []
        boxes: list[list[int]] = []
        labels: list[int] = []

        for line in valid_lines:
            category = (line.get("category") or "").upper()
            line_words = line.get("words", [])

            b_label = f"B-{category}"
            i_label = f"I-{category}"

            for w_idx, word_obj in enumerate(line_words):
                text = (word_obj.get("text") or "").strip()
                if not text:
                    continue

                box = self._quad_to_bbox(word_obj.get("quad", {}), img_w, img_h)

                tag = b_label if w_idx == 0 else i_label
                # Categories outside our label set get folded into O so
                # training stays well-defined.
                label_id = self.label2id.get(tag, o_id)

                words.append(text)
                boxes.append(box)
                labels.append(label_id)

        return words, boxes, labels

    @staticmethod
    def _quad_to_bbox(quad: dict, img_w: int, img_h: int) -> list[int]:
        """
        CORD stores quads as 4 (x, y) corner points. Reduce to an
        axis-aligned bounding box, then rescale to 0-1000 for LayoutLMv3.
        """
        if not quad:
            return [0, 0, 0, 0]

        xs = [quad.get(f"x{i}", 0) for i in range(1, 5)]
        ys = [quad.get(f"y{i}", 0) for i in range(1, 5)]
        x0, x1 = min(xs), max(xs)
        y0, y1 = min(ys), max(ys)

        # Normalize to LayoutLMv3's 0-1000 grid and clamp.
        def norm(v, full):
            if full <= 0:
                return 0
            return max(0, min(1000, int(1000 * v / full)))

        return [
            norm(x0, img_w),
            norm(y0, img_h),
            norm(x1, img_w),
            norm(y1, img_h),
        ]
