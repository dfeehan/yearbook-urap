# Layout Parser vs Face Detection for Yearbook Processing

This document compares deep learning layout detection ([Layout-Parser](https://layout-parser.github.io/)) with classical computer vision (face detection) for yearbook photo extraction.

## Key Finding: Face Detection Works Better for Yearbooks

**Deep Learning (Layout-Parser):** Detects 5 huge regions (treats entire grid as one "Figure")
**Face Detection:** Successfully detects 24 individual student photos

## Why Layout-Parser Fails for Yearbooks

Layout-Parser models are trained on PubLayNet dataset (academic papers with text, tables, figures). They detect **semantic regions** (what type of content), not **geometric patterns** (photo grids).

For yearbook grids: Use **`yearbook_grid_detector.py`** instead - combines face detection with grid inference.

## Comparison Results

Tested on `185777725.jpg`:
- **EfficientDet (Deep Learning)**: 5 regions detected
  - 2 massive "Figure" regions covering entire page
  - 3 small "Title" text regions
  - Cannot identify individual student photos
  
- **Face Detection + Grid Analysis**: 24 individual photos detected
  - 14 faces detected directly
  - 10 additional photos inferred from grid pattern
  - Successfully crops all student photos

## Recommended Solution: Face Detection

Use `yearbook_grid_detector.py` for automatic yearbook photo extraction:

```bash
python code/images/yearbook_grid_detector.py \
  --input yearbook_page.jpg \
  --output results/
```

**Output:**
- `*_grid_detection.png` - Visualization with all 24 photos highlighted
- `*_crops/` - Individual student photo crops
- `*_metadata.json` - Grid coordinates and statistics

**How it works:**
1. Haar Cascade face detection finds initial photos
2. Grid analysis infers photo positions from detected patterns
3. Completes grid by finding missing photos based on spacing

## Layout-Parser Documentation (For Reference)

### Installation

```bash
pip install "layoutparser[ocr]" "layoutparser[effdet]" opencv-python
```

### Test Deep Learning Detection

```bash
# Compare deep learning vs face detection
python code/images/test_detectron2_approach.py \
  --input yearbook_page.jpg \
  --output comparison/ \
  --confidence 0.3
```

### Basic Layout Detection

```bash
python code/images/layout_parser_demo.py \
  --input yearbook_page.jpg \
  --output layout_results/
```

**Note:** This will detect large regions, not individual photos.
python code/images/layout_parser_demo.py --input page.jpg --confidence 0.3

# More strict (only high-confidence detections)
python code/images/layout_parser_demo.py --input page.jpg --confidence 0.7
```

## Available Models

| Model | Speed | Accuracy | Best For |
|-------|-------|----------|----------|
| `lp://PubLayNet/tf_efficientdet_d0` | Fast | Good | General documents (default) |
| `lp://PubLayNet/tf_efficientdet_d1` | Medium | Better | More accurate detection |
| `lp://NewspaperNavigator/faster_rcnn_R_50_FPN_3x` | Slow | Best | Historical documents |

Example with historical document model:
```bash
python code/images/layout_parser_demo.py \
  --input yearbook_page.jpg \
  --model lp://NewspaperNavigator/faster_rcnn_R_50_FPN_3x
```

## Output Format

### Metadata JSON
```json
{
  "source_image": "yearbook_page.jpg",
  "num_regions": 12,
  "regions": [
    {
      "id": 0,
      "type": "Figure",
      "confidence": 0.95,
      "bbox": {
        "x1": 100.5,
### Recommended: Use Face Detection

Replace Label Studio manual annotations with automatic face detection:

1. **Automatic extraction**: `yearbook_grid_detector.py` finds all student photos
2. **No manual work**: Eliminates need for Label Studio annotations
3. **Grid completion**: Infers missing photos from detected pattern
4. **Direct to OCR**: Crops ready for text extraction pipeline
      }
    }
  ]
}
```

## Integration with Existing Pipeline

The detected regions can replace or supplement your Label Studio annotations:

1. **Replace manual annotation**: Use layout-parser to automatically find all photo regions
2. **Pre-annotation**: Use layout-parser to create initial bounding boxes, then refine in Label Studio
3. **Validation**: Check that manual annotations match automatic detections

## Region Types Detected

- **Figure**: Student photos, images
- **Text**: Name labels, captions, body text
- **Title**: Page headers, section titles
- **Table**: Organized data (if present)
- **List**: Enumerated items

## Tips for Best Results

1. **Start with default settings** - The base model works well for most yearbooks
2. **Adjust confidence** - Lower for older/faded yearbooks, higher for clean modern ones
3. **Try historical model** - If yearbooks are pre-1960s, use NewspaperNavigator model
4. **Post-process** - Filter detected regions by size/aspect ratio to find just photos
1. **Use `yearbook_grid_detector.py`** for production photo extraction
2. **Integrate with URAPCrop.py** to replace Label Studio annotations
3. **Test on your full yearbook dataset** to validate grid detection
4. **Optional**: Use layout-parser only for non-grid layouts (text pages, covers)ipeline
2. Replace or augment Label Studio workflow
3. Add OCR directly to detected text regions
4. Build automatic name-photo matching

## Troubleshooting

**No regions detected:**
### Layout-Parser Issues

**Not detecting individual photos:**
- This is expected for yearbook grids
- Use `yearbook_grid_detector.py` instead

**Slow processing:**
- Face detection (Haar Cascades) is much faster than deep learning
- No GPU required for classical CV approach

### Face Detection Issues

**Missing photos:**
- Check `--min-neighbors` parameter (lower = more sensitive)
- Adjust `--scale-factor` for different photo sizes
- Grid inference will complete missing photos automatically

**False positives:**
- Increase `--min-neighbors` parameter
- Add size filtering in post-