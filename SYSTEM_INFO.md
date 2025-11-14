# System Hardware Specifications

**Test Environment:**

- **Model**: MacBook Pro (MacBookPro17,1)
- **Chip**: Apple M1
- **CPU Cores**: 8 (4 performance + 4 efficiency)
- **Memory**: 16 GB
- **OS**: macOS

## Benchmark Results

All tests performed on the above hardware configuration.

### OCR Correction (Mistral)
- **Model**: mistral:latest (4.4 GB)
- **Test file**: sample_ocr.txt (571 characters, 24 lines)
- **Processing time**: ~5-10 seconds
- **Chunk size**: 5 lines per request

### Vision Analysis (LLaVA)

#### Text Extraction
- **Model**: llava:latest (4.7 GB)
- **Test image**: henry_j_smith.png (yearbook photo)
- **Output**: 53 characters extracted
- **Processing time**: ~3-5 seconds

#### Smile Detection
- **Model**: llava:latest (4.7 GB)
- **Timeout**: 300 seconds (5 minutes) per image
- **Test results**:
  - henry_j_smith.png: 131.82 seconds (Smiling: yes, Confidence: high)
  - bertha_savage.png: 165.22 seconds (Smiling: uncertain, Confidence: medium)
  - therman_dewey_williamson.png: 132.82 seconds (Smiling: yes, Confidence: medium)
- **Average processing time**: ~143 seconds per image

## Notes

- All models run locally via Ollama
- No internet connection required after model download
- Processing times may vary based on system load
- First run may be slower due to model loading
