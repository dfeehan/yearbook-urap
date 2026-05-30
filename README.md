# Yearbook URAP

Digitizing, structuring, and understanding 80+ years of historical American university yearbooks using computer vision, OCR, and large language models.

UC Berkeley D-Lab · URAP Research · 2025–Present

---

## What This Project Does

Historical yearbooks contain rich records of student demographics, activities, and culture — but exist only as scanned images, unsearchable and unstructured. This project builds an end-to-end pipeline to automatically:

1. **Detect** every portrait photo on a yearbook page using a custom-trained object detector
2. **Extract** each student's name, activities, quote, and biography using vision LLMs
3. **Digitize** the surrounding text via OCR with LLM-based error correction
4. **Structure** all data into clean JSON for downstream research use

---

## Portrait Detection Model

A custom Faster R-CNN (ResNet-50 FPN) model fine-tuned on 400+ hand-annotated historical yearbook pages.

| Metric | Value |
|---|---|
| Detection accuracy (AP50) | **70%** |
| Pretrained SOTA baseline (AP50) | 41% |
| Improvement | **+71% relative** |
| Model size | **315 MB** |
| Matches Claude-opus-4-7 accuracy | ✓ (60/60 portraits on test set) |

**Model weights are included in this repo via Git LFS** — clone normally and they download automatically.

```
layout-model-training/outputs/yearbook-clean60/fast_rcnn_R_50_FPN/
├── model_final.pth   # trained weights (315 MB, via Git LFS)
└── config.yaml       # Detectron2 config
```

---

## Repository Structure

```
├── code/
│   ├── images/                    # Crop student photos from Label Studio coordinates
│   │   └── URAPCrop.py
│   ├── labelstudio/               # Label Studio API utilities
│   │   └── api.py
│   ├── OCR/                       # OCR digitization pipeline
│   │   ├── OCR_yearbook_pages.py      # Run Tesseract on yearbook page images
│   │   ├── ocr_correction/
│   │   │   └── llm_ocr_correction.py  # Mistral LLM error correction
│   │   └── relationship_views/        # Visualize OCR region relationships
│   └── numident-explore/          # Numident record matching notebooks
│
├── layout-model-training/
│   ├── experiments/
│   │   ├── portrait_text_matcher.py   # Full pipeline: detector → GPT-4o → JSON + table
│   │   ├── vision_only_matcher.py     # Baseline: Claude-opus-4-7 with no detector
│   │   └── portrait_matches/          # Demo outputs (JSON + table PNGs)
│   ├── configs/yearbook/              # Detectron2 training configs
│   ├── data/yearbook/                 # Training annotations (images downloaded separately)
│   ├── outputs/yearbook-clean60/      # Best trained model weights (via Git LFS)
│   ├── scripts/                       # SLURM training scripts for NERSC Perlmutter
│   ├── tools/train_net.py             # Detectron2 training entry point
│   ├── fetch_annotations_direct.py    # Export annotations from Label Studio
│   ├── create_small_subset.py         # Create 50-image training subset
│   ├── create_clean_text_training_data.py
│   ├── split_data.py
│   ├── detect_and_label_humans.py
│   ├── clean_duplicate_annotations.py
│   └── test_trained_model.py
│
├── requirements.txt
├── .env.example
└── SYSTEM_INFO.md                 # NERSC Perlmutter environment notes
```

---

## Quick Start

### 1. Install dependencies

```bash
git clone https://github.com/dfeehan/yearbook-urap
cd yearbook-urap
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

> Model weights download automatically via Git LFS on clone. Requires `git-lfs` installed (`brew install git-lfs` / `apt install git-lfs`).

### 2. Run the portrait extraction pipeline

```bash
# Set your CBORG / OpenAI-compatible API key
export OPENAI_API_KEY="your-key-here"
export OPENAI_BASE_URL="https://api.openai.com/v1"  # or your gateway

# Full pipeline: custom detector + GPT-4o → JSON + table PNG
python layout-model-training/experiments/portrait_text_matcher.py \
  --image path/to/yearbook_page.jpg

# Vision-only baseline: Claude-opus-4-7 (no Detectron2 required)
python layout-model-training/experiments/vision_only_matcher.py \
  --image path/to/yearbook_page.jpg \
  --model claude-opus-4-7
```

Both scripts write output to `layout-model-training/experiments/portrait_matches/`:
- `*_matches.json` — structured data per portrait (name, activities, quote, bio)
- `*_table.png` — visual table with portrait crops

### 3. OCR pipeline

```bash
# Extract text from yearbook pages with Tesseract
python code/OCR/OCR_yearbook_pages.py

# Correct OCR errors with local LLM (requires Ollama + Mistral)
python code/OCR/ocr_correction/llm_ocr_correction.py --input output.txt
```

### 4. Retrain the model

```bash
# Export annotations from Label Studio
python layout-model-training/fetch_annotations_direct.py

# Create training subset
python layout-model-training/create_small_subset.py

# Train locally
cd layout-model-training/scripts && bash train_yearbook_small.sh

# Train on NERSC Perlmutter
sbatch layout-model-training/scripts/perlmutter_train_clean60.slurm
```

---

## Demo Results

Sample outputs from `layout-model-training/experiments/portrait_matches/`:

| Image | Portraits detected | Method |
|---|---|---|
| `onlineRandomImage` (1932 senior class) | 10 / 10 | Custom detector + GPT-4o |
| `onlineRandomImage` (1932 senior class) | 10 / 10 | Claude-opus-4-7 vision-only |
| `ayantee1939negr` (HBCU 1939) | 20 / 20 | Custom detector + GPT-4o |
| `shawuniversityjo1939shaw` | 10 / 10 | Custom detector + GPT-4o |
| `snipscuts19401940cent` | 20 / 20 | Custom detector + GPT-4o |

---

## Requirements

- Python 3.10+
- [Detectron2](https://github.com/facebookresearch/detectron2) (for portrait_text_matcher.py)
- Ollama + Mistral (for OCR correction, optional)
- OpenAI-compatible API key (for GPT-4o / Claude extraction)
- See `requirements.txt` for full Python dependencies

---

## License

Academic use only.
