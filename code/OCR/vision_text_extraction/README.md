# Vision-Based Text Extraction

Extract text from images using LLaVA vision-language model (experimental).

## What It Does

Uses LLaVA vision model to read and extract text directly from images without traditional OCR preprocessing.

## Requirements

- Ollama with LLaVA model: `ollama pull llava`

## Usage

```bash
python llava_vision_ocr.py --input input/image.jpg --output output/extracted.txt
```

Process multiple images:
```bash
python llava_vision_ocr.py --input input/ --output output/
```

## Performance

- **Processing time**: ~100-300 seconds per image (Apple M1)
- **Accuracy**: Variable - works better for simple text, may hallucinate on complex documents

## ⚠️ Limitations

Vision models are **not reliable** for accurate text extraction from historical yearbooks:
- May hallucinate completely unrelated text
- Non-deterministic (different results each run)
- Much slower than traditional OCR
- Better suited for image understanding, not precise text reading

**Recommendation**: Use traditional OCR (`ocr_correction/`) for text extraction. Use vision models only for experimental purposes or image analysis tasks.

## Output

Plain text files containing extracted text (with caveats above).
