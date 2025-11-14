# Smile Detection

Analyze facial expressions in yearbook photos using LLaVA vision model.

## What It Does

Detects whether students are smiling in yearbook photos with confidence levels and explanations.

## Requirements

- Ollama with LLaVA model: `ollama pull llava`

## Usage

Single image:
```bash
python smile_detector.py --input input/student_photo.png --output output/result.json
```

Multiple images:
```bash
python smile_detector.py --input input/ --output output/
```

Custom timeout (default 300 seconds):
```bash
python smile_detector.py --input input/photo.png --timeout 180
```

## Input

Place student yearbook photos in the `input/` directory. Three sample students included:
- `henry_j_smith.png`
- `bertha_savage.png`
- `therman_dewey_williamson.png`

## Output

JSON files with:
- **smiling**: yes/no/uncertain
- **confidence**: high/medium/low
- **explanation**: Detailed description of facial expression
- **processing_time_seconds**: Time taken for analysis

## Performance

On Apple M1 MacBook Pro (16GB RAM):
- **Average processing time**: ~100-130 seconds per image
- **Accuracy**: Good for clear facial expressions, may be inconsistent

## Notes

- Results may vary between runs (non-deterministic)
- Model sometimes gives brief responses, re-run if needed
- English language instruction included to prevent multilingual responses
