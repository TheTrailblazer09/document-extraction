import json
from pathlib import Path
from contextlib import contextmanager
from PIL import Image
from datasets import load_from_disk

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = REPO_ROOT / "data"


class DocumentDataset:
    """Class for Document data with train/val/test splits."""

    def __init__(self, dataset_name, data_dir="data"):
        self.dataset_name = dataset_name.lower()
        self.data_dir = self._resolve_data_dir(data_dir)
        self.train = []
        self.val = []
        self.test = []

        loaders = {"cord": self._load_cord, "funsd": self._load_funsd, "sroie": self._load_sroie}
        if self.dataset_name not in loaders:
            raise ValueError(f"Please refactor data_loading.py to work with this new dataset: {dataset_name}")
        loaders[self.dataset_name]()

    def _resolve_data_dir(self, data_dir):
        path = Path(data_dir).expanduser()
        if path.is_absolute():
            return path
        if path.exists():
            return path.resolve()

        repo_relative = (REPO_ROOT / path).resolve()
        if repo_relative.exists():
            return repo_relative

        return repo_relative

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
        sroie_path = self.data_dir / "sroie"
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
        try:
            return load_from_disk(str(path))
        except TypeError as exc:
            if "dataclass type or instance" not in str(exc):
                raise
            self._rewrite_legacy_list_features(path)
            with self._legacy_list_compat():
                return load_from_disk(str(path))

    def _rewrite_legacy_list_features(self, dataset_path):
        updated = False
        for info_path in dataset_path.glob("*/dataset_info.json"):
            data = json.loads(info_path.read_text(encoding="utf-8"))
            rewritten, changed = self._replace_legacy_list_types(data)
            if changed:
                info_path.write_text(json.dumps(rewritten, indent=2) + "\n", encoding="utf-8")
                updated = True
        return updated

    def _replace_legacy_list_types(self, obj):
        changed = False

        if isinstance(obj, dict):
            rewritten = {}
            for key, value in obj.items():
                if key == "_type" and value == "List":
                    rewritten[key] = "Sequence"
                    changed = True
                else:
                    rewritten_value, child_changed = self._replace_legacy_list_types(value)
                    rewritten[key] = rewritten_value
                    changed = changed or child_changed
            return rewritten, changed

        if isinstance(obj, list):
            rewritten = []
            for value in obj:
                rewritten_value, child_changed = self._replace_legacy_list_types(value)
                rewritten.append(rewritten_value)
                changed = changed or child_changed
            return rewritten, changed

        return obj, False

    @contextmanager
    def _legacy_list_compat(self):
        from datasets.features import features as dataset_features

        original_list = getattr(dataset_features, "List", None)
        dataset_features.List = dataset_features.Sequence
        try:
            yield
        finally:
            dataset_features.List = original_list


def load_dataset(dataset_name, data_dir="data"):
    """
    Load a dataset by name. Returns dataset with .train / .val / .test, 
    val should be empty for funsd and sroie.
    """
    return DocumentDataset(dataset_name, data_dir)
