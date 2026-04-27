import json
from pathlib import Path
from PIL import Image
from datasets import load_from_disk


class DocumentDataset:
    """Class for Document data with train/val/test splits."""

    def __init__(self, dataset_name, data_dir="data"):
        self.dataset_name = dataset_name.lower()
        self.data_dir = Path(data_dir)
        self.train = []
        self.val = []
        self.test = []

        loaders = {"cord": self._load_cord, "funsd": self._load_funsd, "sroie": self._load_sroie}
        if self.dataset_name not in loaders:
            raise ValueError(f"Please refactor data_loading.py to work with this new dataset: {dataset_name}")
        loaders[self.dataset_name]()

    """
    CORD - HF
    fields: image, ground_truth
    """
    def _load_cord(self):
        dataset = self._hf_load("cord")
        for item in dataset["train"]:
            self.train.append({
                "image": item["image"],
                "ground_truth": item["ground_truth"],
                "datasetID":    "cord"})
        for item in dataset["validation"]:
            self.val.append({
                "image": item["image"],
                "ground_truth": item["ground_truth"],
                "datasetID": "cord"})
        for item in dataset["test"]:
            self.test.append({
                "image": item["image"],
                "ground_truth": item["ground_truth"],
                "datasetID": "cord"})

        print(f"Loaded CORD:  {len(self.train)} train, {len(self.val)} val, {len(self.test)} test")

    """
    FUNSD - HF
    fields: image, tokens, bboxes, ner_tags
    """
    def _load_funsd(self):
        dataset = self._hf_load("funsd")
        for item in dataset["train"]:
            self.train.append({
                "image": item["image"],
                "tokens": item["tokens"],
                "bboxes": item["bboxes"],
                "ner_tags": item["ner_tags"],
                "datasetID": "funsd",
            })
        for item in dataset["test"]:
            self.test.append({
                "image": item["image"], "tokens": item["tokens"],
                "bboxes": item["bboxes"], "ner_tags": item["ner_tags"],
                "datasetID": "funsd",
            })
        print(f"Loaded FUNSD: {len(self.train)} train, {len(self.val)} val, {len(self.test)} test")

    """
    SROIE
    fields: image, boxes, entities
    img/   *.jpg
    box/   *.txt  — lines of  x1,y1,...,x4,y4,text
    entities/  *.txt  — JSON  {company, date, address, total}
    """
    def _load_sroie(self):
        sroie_path = self.data_dir / "sroie" / "hf_dataset"
        if not sroie_path.exists():
            raise FileNotFoundError(f"SROIE dataset not found at {sroie_path}")

        for split_name, split_list in [("train", self.train), ("test", self.test)]:
            img_dir      = sroie_path / split_name / "img"
            box_dir      = sroie_path / split_name / "box"
            entities_dir = sroie_path / split_name / "entities"

            for img_file in sorted(img_dir.glob("*.jpg")):
                stem = img_file.stem

                boxes = []
                for line in (box_dir / f"{stem}.txt").read_text(encoding="latin-1").strip().splitlines():
                    parts = line.split(",", 8)
                    if len(parts) == 9:
                        boxes.append({"coords": list(map(int, parts[:8])), "text": parts[8]})

                entities = json.loads((entities_dir / f"{stem}.txt").read_text(encoding="latin-1"))

                split_list.append({
                    "image": Image.open(img_file).copy(),
                    "boxes": boxes,
                    "entities": entities,
                    "datasetID": "sroie",
                })

        print(f"Loaded SROIE: {len(self.train)} train, {len(self.val)} val, {len(self.test)} test")

    """
    Shared HuggingFace loader, sroie is from kaggle
    """
    def _hf_load(self, name):
        path = self.data_dir / name / "hf_dataset"
        if not path.exists():
            raise FileNotFoundError(f"{name.upper()} dataset not found at {path}")
        return load_from_disk(str(path))


def load_dataset(dataset_name, data_dir="data"):
    """
    Load a dataset by name. Returns dataset with .train / .val / .test,
    val should be empty for funsd and sroie.
    """
    return DocumentDataset(dataset_name, data_dir)
