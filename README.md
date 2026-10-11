# Comparing Layout-Aware Transformers for Document Information Extraction

## Project Overview
Comparing LayoutLMv3, Donut, and an OCR+CNN baseline for structured information extraction from financial document images. Evaluated on CORD, FUNSD, and SROIE datasets.

## Setup

### Prerequisites
- Docker and Docker Compose
- NVIDIA Container Toolkit (for GPU training only)

### 1. Clone the repo
```bash
git clone https://github.com/baileyseyller/doc-extraction-project.git
cd doc-extraction-project
```

### 2. Build the container
```bash
# GPU (for training)
docker compose build gpu

# CPU only (for data work, evaluation, writing code)
docker compose build cpu
```

### 3. Download datasets
```bash
docker compose run --rm cpu python scripts/download_data.py
```

### 4. Verify setup
```bash
docker compose run --rm cpu python scripts/verify_setup.py
```

### Common commands
```bash
# Interactive shell with GPU
docker compose run --rm gpu bash

# Interactive shell CPU only
docker compose run --rm cpu bash

# Start Jupyter notebooks
docker compose up notebook
# then open http://localhost:8888

# Fine-tune LayoutLMv3 on CORD
docker compose run --rm gpu python -m src.training.train --config configs/layoutlmv3.yaml --dataset cord

# Fine-tune Donut on CORD
docker compose run --rm gpu python -m src.training.train --config configs/donut.yaml --dataset cord

# Run evaluation
docker compose run --rm cpu python -m src.evaluation.evaluate --model layoutlmv3 --dataset cord --checkpoint models/layoutlmv3_cord_best.pt

# Run shared Donut evaluation and save a normalized results JSON
docker compose run --rm cpu python -m src.evaluation.evaluate --model donut --dataset cord --checkpoint models/donut_cord_best.pt --noise-setting clean

# Aggregate normalized result JSONs into report tables
docker compose run --rm cpu python scripts/analyze_results.py
```

### No GPU?
The CPU container works for everything except training. Use it for data exploration, writing code, running evaluation on saved checkpoints, and notebooks. For actual training runs, use Colab Pro.

## Project Structure
```
.
|-- Dockerfile              # GPU container (CUDA + PyTorch)
|-- Dockerfile.cpu          # CPU container (lighter, for dev work)
|-- docker-compose.yaml     # Orchestration for gpu/cpu/notebook services
|-- configs/                # Training configs (hyperparams, model settings)
|   |-- layoutlmv3.yaml
|   |-- donut.yaml
|   +-- baseline_cnn.yaml
|-- data/                   # Downloaded datasets (gitignored, mounted as volume)
|   |-- cord/
|   |-- funsd/
|   +-- sroie/
|-- models/                 # Saved model checkpoints (gitignored, mounted as volume)
|-- notebooks/              # Exploration, visualization, analysis
|   |-- 01_data_exploration.ipynb
|   |-- 02_results_analysis.ipynb
|   +-- 03_error_analysis.ipynb
|-- results/                # Experiment logs, metrics, plots
|-- docs/                   # Report scaffolding and shared writeup notes
|-- scripts/                # Standalone scripts (download, preprocess, etc.)
|   |-- download_data.py
|   +-- verify_setup.py
|-- src/                    # Main source code
|   |-- data/               # Dataset loading, preprocessing, augmentation
|   |-- models/             # Model definitions and wrappers
|   |-- training/           # Training loops, fine-tuning logic
|   |-- evaluation/         # Metrics, inference, analysis
|   +-- utils/              # Shared helpers
|-- requirements.txt
|-- .gitignore
+-- README.md
```

## Experiments

| # | Experiment | Owner | Status |
|---|-----------|-------|--------|
| 1 | LayoutLMv3 fine-tuning (CORD + FUNSD) | | TODO |
| 2 | Donut fine-tuning (CORD + FUNSD) | | TODO |
| 3 | OCR + CNN baseline | | TODO |
| 4 | Layout ablation (remove 2D embeddings) | | TODO |
| 5 | Noise robustness testing | | TODO |
| 6 | Cross-domain transfer (stretch) | | TODO |
| 7 | Pretrained vs scratch (stretch) | | TODO |

## Results
Results and analysis will be documented in `results/` and in the final report.
