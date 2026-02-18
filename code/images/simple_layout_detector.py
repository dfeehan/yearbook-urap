"""
Simple Layout Detection using OpenCV

A lightweight alternative to layout-parser that uses OpenCV's contour detection
to find bounding boxes for photos and text regions on yearbook pages.

No external models needed - works offline!

Usage:
    python code/images/simple_layout_detector.py --input yearbook_page.jpg
"""

import argparse
import cv2
import numpy as np
from PIL import Image, ImageDraw
from pathlib import Path
import json


def detect_regions_opencv(image_path, min_area=5000, max_area=None, aspect_ratio_range=(0.5, 2.5)):
    """
    Detect rectangular regions in a yearbook page using OpenCV.
    
    Args:
        image_path: Path to yearbook image
        min_area: Minimum region area (pixels)
        max_area: Maximum region area (None = image size / 10)
        aspect_ratio_range: (min, max) width/height ratios to accept
        
    Returns:
        regions: List of detected bounding boxes
        image: Original image
    """
    print(f"\n{'='*60}")
    print(f"Processing: {Path(image_path).name}")
    print(f"{'='*60}")
    
    # Load image
    img_color = cv2.imread(str(image_path))
    if img_color is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    h, w = img_color.shape[:2]
    if max_area is None:
        max_area = (w * h) // 10  # Max 10% of image
    
    print(f"Image size: {w} x {h}")
    
    # Convert to grayscale
    gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    
    # Multiple detection strategies
    regions = []
    
    # Strategy 1: Edge detection
    print("Detecting edges...")
    edges = cv2.Canny(gray, 50, 150)
    edges_dilated = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=2)
    contours1, _ = cv2.findContours(edges_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Strategy 2: Thresholding
    print("Applying adaptive threshold...")
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)
    thresh_dilated = cv2.dilate(thresh, np.ones((5, 5), np.uint8), iterations=2)
    contours2, _ = cv2.findContours(thresh_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Combine contours from both strategies
    all_contours = list(contours1) + list(contours2)
    
    print(f"Found {len(all_contours)} raw contours")
    
    # Filter and extract bounding boxes
    seen_regions = set()
    
    for contour in all_contours:
        # Get bounding box
        x, y, w_box, h_box = cv2.boundingRect(contour)
        area = w_box * h_box
        
        # Filter by area
        if area < min_area or area > max_area:
            continue
        
        # Filter by aspect ratio (avoid very thin boxes)
        aspect_ratio = w_box / h_box if h_box > 0 else 0
        if aspect_ratio < aspect_ratio_range[0] or aspect_ratio > aspect_ratio_range[1]:
            continue
        
        # Avoid duplicates (similar boxes)
        region_key = (x // 20, y // 20, w_box // 20, h_box // 20)  # Grid quantization
        if region_key in seen_regions:
            continue
        seen_regions.add(region_key)
        
        # Determine type based on characteristics
        region_type = classify_region(img_color[y:y+h_box, x:x+w_box], aspect_ratio)
        
        regions.append({
            'bbox': (x, y, x + w_box, y + h_box),
            'width': w_box,
            'height': h_box,
            'area': area,
            'aspect_ratio': aspect_ratio,
            'type': region_type
        })
    
    # Sort by position (top to bottom, left to right)
    regions.sort(key=lambda r: (r['bbox'][1] // 100, r['bbox'][0]))
    
    print(f"\nDetected {len(regions)} regions:")
    for i, region in enumerate(regions):
        bbox = region['bbox']
        print(f"  {i+1}. Type: {region['type']:10s} | "
              f"Size: {region['width']}x{region['height']} | "
              f"Position: ({bbox[0]}, {bbox[1]})")
    
    return regions, img_color


def classify_region(region_img, aspect_ratio):
    """
    Classify a region as photo, text, or title based on its characteristics.
    
    Args:
        region_img: Cropped region image
        aspect_ratio: Width/height ratio
        
    Returns:
        type: 'Photo', 'Text', or 'Title'
    """
    if region_img.size == 0:
        return 'Unknown'
    
    # Convert to grayscale for analysis
    gray = cv2.cvtColor(region_img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    
    # Calculate edge density
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / (w * h)
    
    # Calculate variance (photos typically have higher variance)
    variance = np.var(gray)
    
    # Classification heuristics
    if aspect_ratio > 1.5 and edge_density < 0.1:
        return 'Title'  # Wide, low edge density = horizontal text
    elif variance > 1000 and aspect_ratio > 0.7 and aspect_ratio < 1.3:
        return 'Photo'  # High variance, squarish = likely photo
    elif edge_density > 0.15:
        return 'Photo'  # High edge density = likely photo
    else:
        return 'Text'  # Default to text


def visualize_regions(image, regions, output_path):
    """
    Create visualization with detected regions highlighted.
    """
    # Convert BGR to RGB for PIL
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_image, 'RGBA')
    
    # Color mapping
    colors = {
        'Photo': (0, 255, 0, 60),    # Green
        'Text': (255, 0, 0, 60),     # Red
        'Title': (0, 0, 255, 60),    # Blue
        'Unknown': (128, 128, 128, 60)  # Gray
    }
    
    for region in regions:
        x1, y1, x2, y2 = region['bbox']
        color = colors.get(region['type'], colors['Unknown'])
        
        # Draw rectangle
        draw.rectangle([x1, y1, x2, y2], fill=color, outline=color[:3] + (255,), width=2)
        
        # Draw label
        label = f"{region['type']}"
        draw.text((x1 + 5, y1 + 5), label, fill=(255, 255, 255, 255))
    
    pil_image.save(output_path)
    print(f"\n✅ Visualization saved to: {output_path}")


def extract_regions(image, regions, output_dir):
    """
    Extract and save individual regions.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nExtracting regions to: {output_dir}")
    
    for i, region in enumerate(regions):
        x1, y1, x2, y2 = region['bbox']
        
        # Crop region
        cropped = image[y1:y2, x1:x2]
        
        # Save
        filename = f"region_{i:03d}_{region['type'].lower()}_{region['width']}x{region['height']}.png"
        output_path = output_dir / filename
        
        cv2.imwrite(str(output_path), cropped)
        print(f"  Saved: {filename}")


def export_metadata(regions, image_path, output_path):
    """
    Export detected regions as JSON metadata.
    """
    metadata = {
        'source_image': str(image_path),
        'num_regions': len(regions),
        'regions': []
    }
    
    for i, region in enumerate(regions):
        x1, y1, x2, y2 = region['bbox']
        metadata['regions'].append({
            'id': i,
            'type': region['type'],
            'bbox': {
                'x1': int(x1),
                'y1': int(y1),
                'x2': int(x2),
                'y2': int(y2),
                'width': int(region['width']),
                'height': int(region['height'])
            },
            'area': int(region['area']),
            'aspect_ratio': float(region['aspect_ratio'])
        })
    
    with open(output_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✅ Metadata saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Simple layout detection using OpenCV (no ML models needed)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic detection
  python code/images/simple_layout_detector.py --input yearbook_page.jpg
  
  # Adjust sensitivity
  python code/images/simple_layout_detector.py --input page.jpg --min-area 3000
  
  # Custom output directory
  python code/images/simple_layout_detector.py --input page.jpg --output results/

Tips:
  - Lower --min-area to detect smaller regions (names, captions)
  - Adjust --min-area if you're missing photos or getting too many false positives
  - This works completely offline - no model downloads needed!
        """
    )
    
    parser.add_argument('--input', required=True, help='Input yearbook image')
    parser.add_argument('--output', default='opencv_layout_output',
                       help='Output directory (default: opencv_layout_output)')
    parser.add_argument('--min-area', type=int, default=5000,
                       help='Minimum region area in pixels (default: 5000)')
    parser.add_argument('--max-area', type=int, default=None,
                       help='Maximum region area in pixels (default: auto)')
    
    args = parser.parse_args()
    
    # Process image
    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Detect regions
    regions, image = detect_regions_opencv(input_path, args.min_area, args.max_area)
    
    if len(regions) == 0:
        print("\n⚠️  No regions detected. Try lowering --min-area")
        return
    
    # Create outputs
    base_name = input_path.stem
    
    # 1. Visualization
    viz_path = output_dir / f"{base_name}_detected.png"
    visualize_regions(image, regions, viz_path)
    
    # 2. Extract regions
    regions_dir = output_dir / f"{base_name}_regions"
    extract_regions(image, regions, regions_dir)
    
    # 3. Metadata
    json_path = output_dir / f"{base_name}_metadata.json"
    export_metadata(regions, input_path, json_path)
    
    print(f"\n{'='*60}")
    print("✅ Processing complete!")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
