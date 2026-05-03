import json
import pytesseract
import torch
from PIL import ImageOps

# Each dataset has its own labels. num_classes must match when building
# the CNN head (pass CORD_NUM_LABELS / FUNSD_NUM_LABELS / SROIE_NUM_LABELS).

# FUNSD — 7 classes
FUNSD_LABELS = [
    "O",
    "B-HEADER", "I-HEADER",
    "B-QUESTION", "I-QUESTION",
    "B-ANSWER", "I-ANSWER",
]
FUNSD_LABEL2ID = {l: i for i, l in enumerate(FUNSD_LABELS)}
FUNSD_NUM_LABELS = len(FUNSD_LABELS)  # 7

# CORD — 33 classes
CORD_LABELS = [
    "O",
    "menu.nm", "menu.num", "menu.unitprice", "menu.cnt",
    "menu.discountprice", "menu.price", "menu.itemsubtotal",
    "menu.vatyn", "menu.etc",
    "menu.sub_nm", "menu.sub_unitprice", "menu.sub_cnt",
    "menu.sub_discountprice", "menu.sub_price", "menu.sub_etc",
    "sub_total.subtotal_price", "sub_total.discount_price",
    "sub_total.service_price", "sub_total.othersvc_price",
    "sub_total.tax_price", "sub_total.tax_and_service", "sub_total.etc",
    "total.total_price", "total.total_etc", "total.cashprice",
    "total.changeprice", "total.creditcardprice", "total.emoneyprice",
    "total.menutype_cnt", "total.menuqty_cnt",
    "void_menu.nm", "void_menu.price",
]
CORD_LABEL2ID = {l: i for i, l in enumerate(CORD_LABELS)}
CORD_NUM_LABELS = len(CORD_LABELS)  # 33

# SROIE — 9 classes
SROIE_LABELS = [
    "O",
    "B-COMPANY", "I-COMPANY",
    "B-DATE",    "I-DATE",
    "B-ADDRESS", "I-ADDRESS",
    "B-TOTAL",   "I-TOTAL",
]
SROIE_LABEL2ID = {l: i for i, l in enumerate(SROIE_LABELS)}
SROIE_NUM_LABELS = len(SROIE_LABELS)  # 9

NUM_LABELS = {
    "cord":  CORD_NUM_LABELS,
    "funsd": FUNSD_NUM_LABELS,
    "sroie": SROIE_NUM_LABELS,
}

# OCR processing

def preprocess(image):
    return ImageOps.grayscale(image)

def run_ocr(image):
    """Return (words, pixel_boxes, confs) from Tesseract. Filters conf < 50."""
    img = preprocess(image)
    data = pytesseract.image_to_data(
        img,
        output_type=pytesseract.Output.DICT,
        config="--oem 3 --psm 6",
    )
    words, boxes, confs = [], [], []
    for i, w in enumerate(data["text"]):
        try:
            conf = float(data["conf"][i])
        except (TypeError, ValueError):
            conf = -1.0

        if w.strip() and conf > 50:
            x, y, w_, h_ = (
                data["left"][i], data["top"][i],
                data["width"][i], data["height"][i],
            )
            words.append(w)
            boxes.append([x, y, x + w_, y + h_])
            confs.append(conf)
    return words, boxes, confs

def normalize_boxes(boxes, width, height):
    """Scale pixel boxes to [0, 1]."""
    return [[x1/width, y1/height, x2/width, y2/height]
            for x1, y1, x2, y2 in boxes]

# FUNSD Label assignment

def _iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0.0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)

def assign_labels_funsd(tess_boxes_norm, gt_boxes, gt_labels):
    """
    Match each Tesseract word to the highest-IoU GT word and inherit its label.
    gt_boxes are in FUNSD's [0, 1000] range and are normalized internally.
    Falls back to O if best IoU < 0.3.
    """
    gt_norm = [[x / 1000 for x in b] for b in gt_boxes]
    labels = []
    for tb in tess_boxes_norm:
        best_iou, best_label = 0.0, 0
        for gb, gl in zip(gt_norm, gt_labels):
            s = _iou(tb, gb)
            if s > best_iou:
                best_iou, best_label = s, gl
        labels.append(best_label if best_iou >= 0.3 else 0)
    return labels

# CORD label assignment

def _cord_add(word_to_label, val, label_name):
    if val:
        lid = CORD_LABEL2ID.get(label_name, 0)
        for tok in str(val).split():
            word_to_label[tok.lower()] = lid

def assign_labels_cord(tess_words, ground_truth_str):
    """
    Match each Tesseract word to a CORD NER label by text lookup.
    Go through the full gt_parse tree: menu[*], menu[*].sub[*], sub_total, total, void_menu.
    Words not found in any entity value are labeled O.
    """
    gt_parse = json.loads(ground_truth_str).get("gt_parse", {})
    word_to_label = {}

    for menu in gt_parse.get("menu", []):
        if not isinstance(menu, dict):
            continue
        for key in ["nm", "num", "unitprice", "cnt", "discountprice",
                    "price", "itemsubtotal", "vatyn", "etc"]:
            _cord_add(word_to_label, menu.get(key), f"menu.{key}")
        sub_items = menu.get("sub", [])
        if isinstance(sub_items, dict):
            sub_items = [sub_items]
        for sub in sub_items:
            if not isinstance(sub, dict):
                continue
            for key in ["nm", "unitprice", "cnt", "discountprice", "price", "etc"]:
                _cord_add(word_to_label, sub.get(key), f"menu.sub_{key}")

    sub_total = gt_parse.get("sub_total", {})
    if isinstance(sub_total, dict):
        for key in ["subtotal_price", "discount_price", "service_price",
                    "othersvc_price", "tax_price", "tax_and_service", "etc"]:
            _cord_add(word_to_label, sub_total.get(key), f"sub_total.{key}")

    total = gt_parse.get("total", {})
    if isinstance(total, dict):
        for key in ["total_price", "total_etc", "cashprice", "changeprice",
                    "creditcardprice", "emoneyprice", "menutype_cnt", "menuqty_cnt"]:
            _cord_add(word_to_label, total.get(key), f"total.{key}")

    void_menu = gt_parse.get("void_menu", {})
    if isinstance(void_menu, dict):
        for key in ["nm", "price"]:
            _cord_add(word_to_label, void_menu.get(key), f"void_menu.{key}")

    return [word_to_label.get(w.lower(), 0) for w in tess_words]

# SROIE label assignment 

def assign_labels_sroie(tess_words, entities):
    """
    Match Tesseract words to SROIE entity fields (company/date/address/total).

    Strategy: for each entity field, try to find the entity value as a
    consecutive run of Tesseract tokens (exact lowercased match). Assign
    B- to the first token, I- to the rest. If no consecutive run is found,
    fall back to labeling individual matching tokens (less precise B/I but
    still labeled). Words that match no entity stay O.

    entities: dict with keys "company", "date", "address", "total"
    """
    labels = [0] * len(tess_words)
    tess_lower = [w.lower() for w in tess_words]

    for field in ["company", "date", "address", "total"]:
        val = entities.get(field, "")
        if not val:
            continue

        seq = [t.lower() for t in str(val).split()]
        b_tag = SROIE_LABEL2ID[f"B-{field.upper()}"]
        i_tag = SROIE_LABEL2ID[f"I-{field.upper()}"]
        n = len(seq)

        # Try consecutive sequence match first
        matched = False
        for start in range(len(tess_lower) - n + 1):
            if tess_lower[start:start + n] == seq:
                labels[start] = b_tag
                for j in range(1, n):
                    labels[start + j] = i_tag
                matched = True
                break

        # Fallback: individual token membership, B- on first hit then I-
        if not matched:
            token_set = set(seq)
            in_entity = False
            for i, w in enumerate(tess_lower):
                if w in token_set:
                    labels[i] = b_tag if not in_entity else i_tag
                    in_entity = True
                elif in_entity:
                    in_entity = False

    return labels

# Process each dataset with OCR and label assignment 

def process_dataset(dataset, save_path):
    """
    Run Tesseract on every image, assign per-word NER labels from ground truth,
    and write a JSON list to save_path. Each entry:
      {words, boxes (norm), confs, labels (per-word int), dataset_id}
    """
    out = []
    for item in dataset:
        img = item["image"]
        w, h = img.size
        dataset_id = item.get("datasetID", "")

        words, boxes, confs = run_ocr(img)
        norm_boxes = normalize_boxes(boxes, w, h)

        if dataset_id == "funsd":
            labels = assign_labels_funsd(norm_boxes, item["bboxes"], item["ner_tags"])
        elif dataset_id == "cord":
            labels = assign_labels_cord(words, item["ground_truth"])
        elif dataset_id == "sroie":
            labels = assign_labels_sroie(words, item["entities"])
        else:
            labels = [0] * len(words)

        out.append({
            "words":      words,
            "boxes":      norm_boxes,
            "confs":      confs,
            "labels":     labels,
            "dataset_id": dataset_id,
        })

    with open(save_path, "w") as f:
        json.dump(out, f)
    print(f"Saved {len(out)} items → {save_path}")

# create grids to pass into CNN 

def build_grid(words, boxes, embed, H=32, W=32):
    """
    Place word embeddings on a (C, H, W) spatial grid.
    Each word fills its entire bounding box region (not just the top-left corner).
    Multiple words overlapping the same cell accumulate with +=.
    """
    grid = torch.zeros(len(embed.vocab), H, W)
    for word, (x1, y1, x2, y2) in zip(words, boxes):
        word = word.lower()                          # vocab is built lowercased
        if word not in embed:
            continue
        vec = embed[word]                            # (C,)
        gx1, gy1 = int(x1 * W), int(y1 * H)
        gx2 = min(max(gx1 + 1, int(x2 * W)), W)
        gy2 = min(max(gy1 + 1, int(y2 * H)), H)
        grid[:, gy1:gy2, gx1:gx2] += vec.unsqueeze(-1).unsqueeze(-1)
    return grid

def build_label_grid(boxes, labels, H=32, W=32):
    """
    Build an (H, W) integer label map for segmentation-style training.
    Each word's bounding box region is filled with its label integer.
    """
    grid = torch.zeros(H, W, dtype=torch.long)
    for (x1, y1, x2, y2), label in zip(boxes, labels):
        gx1, gy1 = int(x1 * W), int(y1 * H)
        gx2 = min(max(gx1 + 1, int(x2 * W)), W)
        gy2 = min(max(gy1 + 1, int(y2 * H)), H)
        grid[gy1:gy2, gx1:gx2] = label
    return grid

# embedding each word for the CNN 

class SimpleEmbed:
    """
    One-hot word embedding over a fixed vocabulary.
    vocab size == number of input channels to the CNN.
    For large vocabularies, swap this for a dense GloVe embedding to keep
    the channel count manageable.
    """
    def __init__(self, vocab):
        self.vocab = {w: i for i, w in enumerate(vocab)}

    def __contains__(self, w):
        return w in self.vocab

    def __getitem__(self, w):
        vec = torch.zeros(len(self.vocab))
        vec[self.vocab[w]] = 1.0
        return vec

# create a pytorch dataset for torch training 

class OCRDataset(torch.utils.data.Dataset):
    """
    Wraps a list of processed items (output of process_dataset).

    __getitem__ returns:
      grid       — (C, H, W) float tensor: word embeddings at spatial positions
      label_grid — (H, W) long tensor:     NER label at each spatial position
    """
    def __init__(self, data, embed, H=32, W=32):
        self.data  = data
        self.embed = embed
        self.H     = H
        self.W     = W

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item       = self.data[idx]
        grid       = build_grid(item["words"], item["boxes"], self.embed, self.H, self.W)
        label_grid = build_label_grid(item["boxes"], item["labels"], self.H, self.W)
        return grid, label_grid
