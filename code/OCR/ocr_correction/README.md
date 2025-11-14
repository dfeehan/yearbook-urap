# OCR Text Correction

Post-process OCR output using local LLM (Mistral) to fix common OCR errors.

## What It Does

Traditional OCR often makes mistakes with similar-looking characters (0/O, 1/l/I, rn/m). This tool uses a local LLM to intelligently correct these errors while preserving the original line structure.

## Requirements

- Ollama installed and running
- Mistral model: `ollama pull mistral`

## Usage

```bash
python llm_ocr_correction.py --input input/sample_ocr.txt --output output/corrected.txt
```

## Input

Place OCR text files in the `input/` directory. See `input/sample_ocr.txt` for an example with common OCR errors.

## Output

Corrected text files are saved to `output/` with the same line structure as the input.

## How It Works

1. Reads OCR text file line by line
2. Identifies potential OCR errors (confusable characters)
3. Sends context to Mistral LLM for intelligent correction
4. Preserves exact line structure (empty lines maintained)
5. Outputs corrected text
