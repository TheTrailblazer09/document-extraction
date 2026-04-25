"""
Dataset classes for Donut fine-tuning on CORD and FUNSD.

Donut doesn't use OCR or bounding boxes -- it takes raw images and produces
JSON token sequences. So the dataset needs to:
1. Load the image
2. Preprocess it for the Swin Transformer encoder
3. Convert the ground truth annotation to a Donut-style token sequence
4. Tokenize that sequence for the BART decoder

Usage:
    from src.data.donut_dataset import DonutDataset

    dataset = DonutDataset(
        dataset_name="cord",
        split="train",
        processor=processor,
        max_length=768,
        task_prompt="<s_cord-v2>",
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
    from albumentations.pytorch import ToTensorV2
    import numpy as np

    ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    ALBUMENTATIONS_AVAILABLE = False


DATA_DIR = Path(__file__).parent.parent.parent / "data"


class DonutDataset(Dataset):
    """
    Dataset for Donut fine-tuning.

    Loads images and ground truth annotations, converts annotations to
    Donut's XML-like token sequences, and tokenizes them for training.
    """

    def __init__(
            self,
            dataset_name: str,
            split: str,
            processor,
            max_length: int = 768,
            task_prompt: str = "<s_cord-v2>",
            augmentation_config: Optional[dict] = None,
    ):
        self.dataset_name = dataset_name
        self.split = split
        self.processor = processor
        self.max_length = max_length
        self.task_prompt = task_prompt
        self.task_end_token = f"</{task_prompt[1:]}"

        # Load the HuggingFace dataset from disk
        self.data = self._load_dataset(dataset_name, split)

        # Set up augmentation if configured
        self.augment = None
        if augmentation_config and augmentation_config.get("enabled", False):
            if not ALBUMENTATIONS_AVAILABLE:
                print(
                    "WARNING: albumentations not installed, "
                    "skipping augmentation"
                )
            else:
                self.augment = A.Compose(
                    [
                        A.Rotate(
                            limit=augmentation_config.get(
                                "rotation_range", 15
                            ),
                            p=0.5,
                        ),
                        A.GaussianBlur(
                            blur_limit=augmentation_config.get(
                                "blur_limit", 5
                            ),
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
        """Load dataset from disk (saved by download_data.py)."""
        from datasets import load_from_disk

        dataset_path = DATA_DIR / dataset_name / "hf_dataset"

        if not dataset_path.exists():
            raise FileNotFoundError(
                f"Dataset not found at {dataset_path}. "
                f"Run: python scripts/download_data.py --dataset {dataset_name}"
            )

        full_dataset = load_from_disk(str(dataset_path))

        # Handle split name differences between datasets
        if split == "val" or split == "validation":
            # CORD has "validation", FUNSD has "test" only
            if "validation" in full_dataset:
                return full_dataset["validation"]
            elif "test" in full_dataset:
                return full_dataset["test"]
            else:
                raise ValueError(
                    f"No validation/test split in {dataset_name}"
                )
        elif split in full_dataset:
            return full_dataset[split]
        else:
            raise ValueError(
                f"Split '{split}' not found. "
                f"Available: {list(full_dataset.keys())}"
            )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        sample = self.data[idx]

        # Get image
        image = self._get_image(sample)

        # Apply augmentation if configured
        if self.augment is not None:
            import numpy as np

            image_np = np.array(image)
            augmented = self.augment(image=image_np)
            image = Image.fromarray(augmented["image"])

        # Get ground truth as token sequence
        gt_json = self._get_ground_truth(sample)
        gt_token_sequence = self._json_to_token_sequence(gt_json)
        full_sequence = (
                self.task_prompt + gt_token_sequence + self.processor.tokenizer.eos_token
        )

        # Process image
        pixel_values = self.processor(
            image, return_tensors="pt"
        ).pixel_values.squeeze(0)

        # Tokenize the target sequence
        tokenized = self.processor.tokenizer(
            full_sequence,
            add_special_tokens=False,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        labels = tokenized.input_ids.squeeze(0)
        # Mask padding positions so they don't contribute to loss
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        return {
            "pixel_values": pixel_values,
            "labels": labels,
        }

    def _get_image(self, sample) -> Image.Image:
        """Extract PIL Image from a dataset sample."""
        if "image" in sample:
            img = sample["image"]
            if isinstance(img, Image.Image):
                return img.convert("RGB")
            # Some HF datasets store images as dicts with path/bytes
            if isinstance(img, dict) and "bytes" in img:
                import io

                return Image.open(io.BytesIO(img["bytes"])).convert("RGB")

        raise ValueError(
            f"Could not extract image from sample. "
            f"Keys: {list(sample.keys())}"
        )

    def _get_ground_truth(self, sample) -> dict:
        """Extract ground truth JSON annotation from a dataset sample."""
        if self.dataset_name == "cord":
            # CORD stores ground truth as a JSON string in "ground_truth"
            gt = sample.get("ground_truth", "")
            if isinstance(gt, str):
                parsed = json.loads(gt)
                # CORD wraps the actual data under "gt_parse"
                return parsed.get("gt_parse", parsed)
            return gt

        elif self.dataset_name == "funsd":
            # FUNSD from nielsr/funsd-layoutlmv3 uses a different format.
            # We need to restructure it into Donut-compatible JSON.
            return self._funsd_to_json(sample)

        elif self.dataset_name == "sroie":
            # SROIE has company, date, address, total as separate fields
            return {
                "company": sample.get("company", ""),
                "date": sample.get("date", ""),
                "address": sample.get("address", ""),
                "total": sample.get("total", ""),
            }

        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")

    def _funsd_to_json(self, sample) -> dict:
        """
        Convert FUNSD's token-level format to Donut-compatible JSON.

        FUNSD has word-level labels (question, answer, header, other).
        We group consecutive words with the same label into entities.
        """
        words = sample.get("words", sample.get("tokens", []))
        ner_tags = sample.get("ner_tags", [])

        # Map tag IDs to names
        tag_names = {0: "other", 1: "question", 2: "answer", 3: "header"}

        entities = {"question": [], "answer": [], "header": [], "other": []}
        current_text = []
        current_tag = None

        for word, tag_id in zip(words, ner_tags):
            tag = tag_names.get(tag_id, "other")
            if tag != current_tag:
                if current_text and current_tag:
                    entities[current_tag].append(" ".join(current_text))
                current_text = [word]
                current_tag = tag
            else:
                current_text.append(word)

        # Don't forget the last group
        if current_text and current_tag:
            entities[current_tag].append(" ".join(current_text))

        # Remove empty categories
        return {k: v for k, v in entities.items() if v}

    def _json_to_token_sequence(self, obj) -> str:
        """Convert a JSON object to Donut's XML-like token sequence."""
        if isinstance(obj, dict):
            tokens = ""
            for key, value in obj.items():
                tokens += f"<s_{key}>"
                tokens += self._json_to_token_sequence(value)
                tokens += f"</s_{key}>"
            return tokens
        elif isinstance(obj, list):
            return "".join(self._json_to_token_sequence(item) for item in obj)
        else:
            return str(obj)


class DonutInferenceDataset(Dataset):
    """
    Minimal dataset for inference only (no labels).
    Takes a list of PIL Images or a directory of image files.
    """

    def __init__(self, images, processor):
        """
        Args:
            images: list of PIL Images, or path to directory of image files
            processor: DonutProcessor instance
        """
        if isinstance(images, (str, Path)):
            img_dir = Path(images)
            self.image_paths = sorted(
                p
                for p in img_dir.iterdir()
                if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
            )
            self.images = None
        else:
            self.images = images
            self.image_paths = None

        self.processor = processor

    def __len__(self) -> int:
        if self.images is not None:
            return len(self.images)
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> dict:
        if self.images is not None:
            image = self.images[idx]
        else:
            image = Image.open(self.image_paths[idx]).convert("RGB")

        pixel_values = self.processor(
            image, return_tensors="pt"
        ).pixel_values.squeeze(0)

        return {"pixel_values": pixel_values}