# Contributing / Work Breakdown

## Team Roles

We have 4 people and 3 models + shared infrastructure. Here's how to split it:

### Alisha: LayoutLMv3 + Project Lead
- Own the LayoutLMv3 fine-tuning pipeline end to end
- Write `src/models/layoutlmv3.py` and `src/training/train_layoutlmv3.py`
- Run LayoutLMv3 experiments on CORD and FUNSD
- Run the layout ablation experiment (disable 2D embeddings, compare)
- Coordinate repo, make sure everyone can run things

### Bailey : Donut
- Own the Donut fine-tuning pipeline end to end
- Write `src/models/donut.py` and `src/training/train_donut.py`
- Run Donut experiments on CORD and FUNSD
- Handle the JSON parsing / tree edit distance evaluation specific to Donut

### David : OCR + CNN Baseline + Data Pipeline
- Build the OCR + CNN baseline model
- Write `src/models/baseline_cnn.py` and `src/training/train_baseline.py`
- Own the shared data loading code (`src/data/`)
- Handle Tesseract OCR preprocessing and bounding box extraction
- Make sure all three models can load data the same way

### Vani: Evaluation + Noise Experiments + Report
- Own the shared evaluation code (`src/evaluation/`)
- Implement noise augmentation pipeline (rotation, blur, resolution reduction)
- Run the noise robustness experiment across all three models
- Run cross-domain transfer experiment if time permits
- Lead the final report writing, generate plots and tables

Everyone contributes to the final report. Person 4 leads it but each person
writes their own model's section.


## Git Workflow

We use feature branches. Never push directly to main.

### Starting new work
```bash
git checkout main
git pull origin main
git checkout -b feature/your-feature-name
```

### When you're done
```bash
git add .
git commit -m "descriptive message about what you did"
git push origin feature/your-feature-name
```
Then open a Pull Request on GitHub. Someone else reviews and merges.

### Branch naming
- `feature/layoutlmv3-training` -- new functionality
- `fix/donut-data-loading` -- bug fixes
- `experiment/noise-robustness` -- experiment-specific branches

### Commit messages
Keep them short and useful:
- "Add LayoutLMv3 fine-tuning loop for CORD"
- "Fix bounding box normalization in data loader"
- "Add noise augmentation transforms"

NOT:
- "stuff"
- "updates"
- "fixed things"


## Timeline

| Week | Milestone |
|------|-----------|
| Week 1 (Mar 24-30) | Repo set up, Docker builds working, data downloaded, everyone can run verify_setup.py |
| Week 2 (Mar 31 - Apr 6) | Data loading pipeline done, each person has their model loading pretrained weights |
| Week 3-4 (Apr 7-20) | Fine-tuning runs complete on CORD for all 3 models |
| Week 5 (Apr 21-27) | Core experiments done (model comparison, layout ablation, noise) |
| Week 6 (Apr 28 - May 4) | Stretch experiments, error analysis, report writing, submit |

Proposal due: Mar 23
Report due: May 4 (grace period May 6)


## Compute

Everything runs inside Docker containers. See README for build/run commands.

Options for GPU training:
1. **Local GPU with Docker** -- if you have an NVIDIA GPU + nvidia-container-toolkit
2**Google Colab Pro** ($10/mo) -- easiest, T4/A100 GPUs (no Docker needed here, just upload the notebook and scripts)
3**Personal GPU without Docker** -- install deps from requirements.txt manually

Donut is the most GPU-hungry (needs ~16GB VRAM for fine-tuning).
LayoutLMv3 can fine-tune on a T4 (16GB) with batch size 4.
The CNN baseline will train on anything.
