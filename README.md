# Yearbook URAP

Historical yearbook image processing and text extraction using AI/ML tools.

## Project Overview

This project processes historical yearbook images to extract student photos and associated text information. It uses Label Studio for annotation and provides AI-powered text extraction and analysis.

## Repository Structure

```
code/
├── images/           # Photo extraction from yearbook pages
│   └── URAPCrop.py   # Extract student photos using Label Studio API
├── labelstudio/      # Label Studio API utilities
│   └── api.py        # API integration
├── OCR/              # OCR and vision-based analysis
│   ├── ocr_correction/     # OCR text correction using local LLM
│   │   ├── input/          # Sample OCR input files
│   │   ├── output/         # Corrected output files
│   │   └── llm_ocr_correction.py
│   └── vision_analysis/    # Vision-based analysis with LLaVA
│       ├── input/          # Test student photos
│       ├── output/         # Analysis results
│       ├── llava_vision_ocr.py  # Image-to-text extraction
│       └── smile_detector.py     # Facial expression analysis
└── numident-explore/ # Additional exploration
```

## Setup

1. Clone the repository
2. Create virtual environment: `python -m venv .venv`
3. Activate: `source .venv/bin/activate`
4. Install dependencies: `pip install -r requirements.txt`
5. Copy `.env.example` to `.env` and configure your API keys

## Usage

### Extract Student Photos
```bash
python code/images/URAPCrop.py
```

### OCR Text Correction
```bash
python code/OCR/ocr_correction/llm_ocr_correction.py --input <file.txt>
```

### Vision Analysis
```bash
# Smile detection
python code/OCR/vision_analysis/smile_detector.py --input <image.png>

# Text extraction from images
python code/OCR/vision_analysis/llava_vision_ocr.py --input <image.png>
```

## Requirements

- Python 3.9+
- Ollama (for local LLM models)
- See `requirements.txt` for Python packages

## License

Academic use only.
