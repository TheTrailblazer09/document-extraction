"""
OCR + CNN baseline for document information extraction.

Runs Tesseract OCR on document images, places word embeddings on a spatial
grid, then uses a fully-convolutional CNN to predict a NER label at every
grid position.

Usage:
    from src.models.baseline_cnn import BaselineCNNWrapper

    # Build vocab from preprocessed OCR data
    import json
    data = json.load(open("data/cache/cord_train.json"))
    vocab = sorted({w for item in data for w in item["words"]})

    model = BaselineCNNWrapper(
        dataset_name="cord",
        vocab=vocab,
        grid_size=(64, 64),
    )

    # Inference on a single image
    from PIL import Image
    image = Image.open("receipt.jpg")
    words, boxes, label_names = model.predict_words(image)

    can plot these for qualitative evaluation, and run IoU for comparison to ground truth

    I tried to match interface from donut :)
"""

import json
import os

import torch
import torch.nn as nn

from src.data.ocr_processing import (
    NUM_LABELS,
    CORD_LABELS, FUNSD_LABELS, SROIE_LABELS,
    SimpleEmbed,
    run_ocr, normalize_boxes,
    build_grid, build_label_grid,
    assign_labels_funsd, assign_labels_cord, assign_labels_sroie,
)

LABEL_NAMES = {
    "cord":  CORD_LABELS,
    "funsd": FUNSD_LABELS,
    "sroie": SROIE_LABELS,
}


# seg CNN model

class SegCNN(nn.Module):
    """
    CNN for spatial NER over a word feature grid.

    Input:  (B, in_channels, H, W)  — spatial word embedding grid
    Output: (B, num_classes, H, W)  — per-position class logits
    """

    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64,  3, padding=1), nn.ReLU(),
            nn.Conv2d(64,          128, 3, padding=1), nn.ReLU(),
            nn.Conv2d(128,         256, 3, padding=1), nn.ReLU(),
        )
        # 1×1 conv maps 256 feature channels to num_classes at every position
        self.head = nn.Conv2d(256, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x))


# ── Wrapper ───────────────────────────────────────────────────────────────────

class BaselineCNNWrapper(nn.Module):
    """
    wrapper class
    """
    def __init__(
        self,
        dataset_name: str = "cord",
        vocab: list = None,
        grid_size: tuple = (64, 64),
    ):
        super().__init__()
        self.dataset_name = dataset_name
        self.grid_size    = tuple(grid_size)
        self.num_classes  = NUM_LABELS[dataset_name]
        self.label_names  = LABEL_NAMES[dataset_name]

        # Word embedding — one-hot over vocab (for now, could add dense sem embedding)
        self.embed = SimpleEmbed(vocab or [])
        in_channels = len(self.embed.vocab)

        self.backbone = SegCNN(in_channels, self.num_classes)
        self.loss_fn  = nn.CrossEntropyLoss()

    @property
    def device(self) -> torch.device:
        return next(self.backbone.parameters()).device

    # prepare the data input by normalzing the boxes and building our grid, or feature map, that the CNN learns on 

    def prepare_input(self, image, item: dict = None) -> dict:
        """
        Run OCR on a PIL image and build the tensors needed for training
        or inference.

        Args:
            image: PIL Image (raw document scan)
            item:  raw dataset dict from data_loading.py — required for
                   training (provides ground truth for label assignment).
                   Pass None for inference.

        Returns dict with:
            grid        — (C, H, W) float: word embeddings at spatial positions
            words       — list[str]: Tesseract-detected words
            boxes       — list[[x1,y1,x2,y2]]: normalized [0,1] boxes
            label_grid  — (H, W) long: NER labels per position (training only)
        """
        H, W = self.grid_size
        w, h = image.size

        words, boxes, confs = run_ocr(image)
        norm_boxes = normalize_boxes(boxes, w, h)
        grid = build_grid(words, norm_boxes, self.embed, H, W)

        result = {"grid": grid, "words": words, "boxes": norm_boxes}

        if item is not None:
            dataset_id = item.get("datasetID", self.dataset_name)
            if dataset_id == "funsd":
                labels = assign_labels_funsd(norm_boxes, item["bboxes"], item["ner_tags"])
            elif dataset_id == "cord":
                labels = assign_labels_cord(words, item["ground_truth"])
            elif dataset_id == "sroie":
                labels = assign_labels_sroie(words, item["entities"])
            else:
                labels = [0] * len(words)

            result["label_grid"] = build_label_grid(norm_boxes, labels, H, W)

        return result

    # forward step 

    def forward(self, grids: torch.Tensor) -> torch.Tensor:
        """grids: (B, C, H, W) -> logits: (B, num_classes, H, W)"""
        return self.backbone(grids)

    def training_step(
        self,
        grids: torch.Tensor,
        label_grids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            grids:       (B, C, H, W) float — word embedding grids
            label_grids: (B, H, W) long     — NER label at each grid position
        """
        logits = self(grids)                    # (B, num_classes, H, W)
        return self.loss_fn(logits, label_grids)

    # predict function given feature grid 

    @torch.no_grad()
    def predict(self, grids: torch.Tensor) -> torch.Tensor:
        """
        grids: (B, C, H, W) → predicted label indices (B, H, W).
        Use predict_words for the full OCR → label pipeline.
        """
        return self(grids).argmax(dim=1)

    @torch.no_grad()
    def predict_words(self, image) -> tuple:
        """
        Full inference pipeline: image → OCR → grid → per-word label names.
        this can be run with predict_words(image)

        Returns:
            words       — list[str]
            boxes       — list[[x1,y1,x2,y2]] normalized
            label_names — list[str]  e.g. ["O", "menu.nm", "menu.price", ...]
        """
        H, W    = self.grid_size
        prep    = self.prepare_input(image)           # no item → no label_grid
        grid    = prep["grid"].unsqueeze(0).to(self.device)
        pred    = self.predict(grid).squeeze(0)       # (H, W)

        label_names = []
        for (x1, y1, *_) in prep["boxes"]:
            gx = min(int(x1 * W), W - 1)
            gy = min(int(y1 * H), H - 1)
            idx = pred[gy, gx].item()
            label_names.append(self.label_names[idx])

        return prep["words"], prep["boxes"], label_names

    # saving and loading utils

    def save_pretrained(self, save_dir: str):
        """Save weights and config"""
        os.makedirs(save_dir, exist_ok=True)
        torch.save(
            self.backbone.state_dict(),
            os.path.join(save_dir, "backbone.pt"),
        )
        config = {
            "dataset_name": self.dataset_name,
            "grid_size":    list(self.grid_size),
            "vocab":        list(self.embed.vocab.keys()),
        }
        with open(os.path.join(save_dir, "config.json"), "w") as f:
            json.dump(config, f)
        print(f"Saved → {save_dir}")

    @classmethod
    def from_pretrained(cls, save_dir: str) -> "BaselineCNNWrapper":
        """Load a previously saved model"""
        with open(os.path.join(save_dir, "config.json")) as f:
            config = json.load(f)
        wrapper = cls(
            dataset_name=config["dataset_name"],
            vocab=config["vocab"],
            grid_size=tuple(config["grid_size"]),
        )
        wrapper.backbone.load_state_dict(
            torch.load(os.path.join(save_dir, "backbone.pt"), map_location="cpu")
        )
        return wrapper
