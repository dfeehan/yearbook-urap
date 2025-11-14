# OCR and Vision Analysis

This directory contains tools for extracting and analyzing text from yearbook images.

## Subdirectories

### `ocr_correction/`
Post-process traditional OCR output using local LLM (Mistral) to fix common errors like character confusion (0/O, 1/l/I, rn/m).
**Recommended for text extraction.**

### `vision_text_extraction/`
Experimental: Extract text directly from images using LLaVA vision model (not recommended - prone to hallucination).

### `vision_smile_detection/`
Analyze facial expressions in yearbook photos using LLaVA vision model.

## Quick Start

Each subdirectory contains its own README with detailed usage instructions.

**OCR Correction (Recommended):**
```bash
python ocr_correction/llm_ocr_correction.py --input ocr_correction/input/sample_ocr.txt
```

**Smile Detection:**
```bash
python vision_smile_detection/smile_detector.py --input vision_smile_detection/input/henry_j_smith.png
```

## Requirements

- Ollama with Mistral and LLaVA models installed
- See main project README for full setup instructions
