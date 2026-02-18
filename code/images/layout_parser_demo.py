"""
Automatic Yearbook Layout Detection using Layout-Parser

This script demonstrates how to use layout-parser to automatically detect
student photos and text regions on yearbook pages, potentially replacing
or augmenting manual Label Studio annotations.

Usage:
    python code/images/layout_parser_demo.py --input yearbook_page.jpg
    python code/images/layout_parser_demo.py --input-folder pages/ --output-folder detected/
"""

import argparse
import layoutparser as lp
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import json
import sys


def detect_layout(image_path, model_name='lp://PubLayNet/tf_efficientdet_d0', confidence_threshold=0.5):
    """
    Detect layout elements in a yearbook page.
    
    Args:
        image_path: Path to yearbook image
        model_name: Pre-trained model to use
        confidence_threshold: Minimum confidence for detection
        
    Returns:
        layout: LayoutParser layout object with detected regions
        image: Original image
    """
    print(f"\n{'='*60}")
    print(f"Processing: {Path(image_path).name}")
    print(f"{'='*60}")
    
    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Convert BGR to RGB for layout-parser
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Load pre-trained model
    print(f"Loading model: {model_name}")
    model = lp.EfficientDetLayoutModel(model_name)
    
    # Detect layout
    print("Detecting layout...")
    layout = model.detect(image_rgb)
    
    # Filter by confidence
    layout = lp.Layout([block for block in layout if block.score >= confidence_threshold])
    
    print(f"\nDetected {len(layout)} regions:")
    for i, block in enumerate(layout):
        print(f"  {i+1}. Type: {block.type:15s} | Confidence: {block.score:.2%} | "
              f"Position: ({int(block.coordinates[0])}, {int(block.coordinates[1])}) - "
              f"({int(block.coordinates[2])}, {int(block.coordinates[3])})")
    
    return layout, image_rgb


def visualize_layout(image, layout, output_path):
    """
    Create a visualization with detected regions highlighted.
    
    Args:
        image: Original image (RGB)
        layout: Detected layout
        output_path: Where to save visualization
    """
    # Convert to PIL for drawing
    pil_image = Image.fromarray(image)
    draw = ImageDraw.Draw(pil_image, 'RGBA')
    
    # Color mapping for different types
    colors = {
        'Text': (255, 0, 0, 80),      # Red
        'Title': (0, 0, 255, 80),     # Blue
        'Figure': (0, 255, 0, 80),    # Green
        'Table': (255, 255, 0, 80),   # Yellow
        'List': (255, 0, 255, 80),    # Magenta
    }
    
    # Draw each detected region
    for block in layout:
        x1, y1, x2, y2 = block.coordinates
        color = colors.get(block.type, (128, 128, 128, 80))
        
        # Draw filled rectangle
        draw.rectangle([x1, y1, x2, y2], fill=color, outline=color[:3] + (255,), width=3)
        
        # Draw label
        label = f"{block.type} ({block.score:.0%})"
        draw.text((x1 + 5, y1 + 5), label, fill=(255, 255, 255, 255))
    
    # Save visualization
    pil_image.save(output_path)
    print(f"\n✅ Visualization saved to: {output_path}")


def extract_regions(image, layout, output_dir):
    """
    Extract and save individual detected regions.
    
    Args:
        image: Original image (RGB)
        layout: Detected layout
        output_dir: Directory to save cropped regions
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nExtracting regions to: {output_dir}")
    
    for i, block in enumerate(layout):
        x1, y1, x2, y2 = map(int, block.coordinates)
        
        # Crop region
        cropped = image[y1:y2, x1:x2]
        
        # Save
        filename = f"region_{i:03d}_{block.type.lower()}_{int(block.score*100)}.png"
        output_path = output_dir / filename
        
        cv2.imwrite(str(output_path), cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR))
        print(f"  Saved: {filename}")


def export_metadata(layout, image_path, output_path):
    """
    Export detected regions as JSON metadata.
    
    Args:
        layout: Detected layout
        image_path: Source image path
        output_path: JSON output path
    """
    metadata = {
        'source_image': str(image_path),
        'num_regions': len(layout),
        'regions': []
    }
    
    for i, block in enumerate(layout):
        x1, y1, x2, y2 = block.coordinates
        metadata['regions'].append({
            'id': i,
            'type': block.type,
            'confidence': float(block.score),
            'bbox': {
                'x1': float(x1),
                'y1': float(y1),
                'x2': float(x2),
                'y2': float(y2),
                'width': float(x2 - x1),
                'height': float(y2 - y1)
            }
        })
    
    with open(output_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✅ Metadata saved to: {output_path}")


def process_single_image(input_path, output_dir, model_name, confidence_threshold):
    """Process a single yearbook page."""
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Detect layout
    layout, image = detect_layout(input_path, model_name, confidence_threshold)
    
    if len(layout) == 0:
        print("\n⚠️  No regions detected. Try lowering --confidence threshold.")
        return
    
    # Create outputs
    base_name = input_path.stem
    
    # 1. Visualization
    viz_path = output_dir / f"{base_name}_detected.png"
    visualize_layout(image, layout, viz_path)
    
    # 2. Extract regions
    regions_dir = output_dir / f"{base_name}_regions"
    extract_regions(image, layout, regions_dir)
    
    # 3. Metadata
    json_path = output_dir / f"{base_name}_metadata.json"
    export_metadata(layout, input_path, json_path)
    
    print(f"\n{'='*60}")
    print("✅ Processing complete!")
    print(f"{'='*60}")


def process_folder(input_folder, output_folder, model_name, confidence_threshold):
    """Process multiple yearbook pages."""
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)
    
    # Find all images
    image_extensions = ['.jpg', '.jpeg', '.png', '.tif', '.tiff']
    images = []
    for ext in image_extensions:
        images.extend(input_folder.glob(f'*{ext}'))
        images.extend(input_folder.glob(f'*{ext.upper()}'))
    
    if not images:
        print(f"❌ No images found in {input_folder}")
        return
    
    print(f"\nFound {len(images)} images to process")
    
    for img_path in sorted(images):
        try:
            img_output_dir = output_folder / img_path.stem
            process_single_image(img_path, img_output_dir, model_name, confidence_threshold)
        except Exception as e:
            print(f"\n❌ Error processing {img_path.name}: {e}")
            continue
    
    print(f"\n{'='*60}")
    print(f"✅ Batch processing complete! Processed {len(images)} images.")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description='Automatic yearbook layout detection using Layout-Parser',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single image
  python code/images/layout_parser_demo.py --input yearbook_page.jpg --output detected/
  
  # Folder of images
  python code/images/layout_parser_demo.py --input-folder pages/ --output-folder results/
  
  # Lower confidence threshold for more detections
  python code/images/layout_parser_demo.py --input page.jpg --confidence 0.3
  
  # Use different model
  python code/images/layout_parser_demo.py --input page.jpg --model lp://PubLayNet/tf_efficientdet_d1

Available Models:
  - lp://PubLayNet/tf_efficientdet_d0 (default, fast)
  - lp://PubLayNet/tf_efficientdet_d1 (more accurate, slower)
  - lp://PubLayNet/faster_rcnn_R_50_FPN_3x
  - lp://NewspaperNavigator/faster_rcnn_R_50_FPN_3x (for historical documents)
        """
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--input', help='Input yearbook image')
    input_group.add_argument('--input-folder', help='Folder containing yearbook images')
    
    # Output options
    parser.add_argument('--output', '--output-folder', dest='output_folder',
                       default='layout_parser_output',
                       help='Output directory (default: layout_parser_output)')
    
    # Model options
    parser.add_argument('--model', 
                       default='lp://PubLayNet/tf_efficientdet_d0',
                       help='Pre-trained model to use (default: PubLayNet/efficientdet_d0)')
    
    parser.add_argument('--confidence', type=float, default=0.5,
                       help='Confidence threshold (0.0-1.0, default: 0.5)')
    
    args = parser.parse_args()
    
    # Process
    if args.input:
        process_single_image(args.input, args.output_folder, args.model, args.confidence)
    else:
        process_folder(args.input_folder, args.output_folder, args.model, args.confidence)


if __name__ == '__main__':
    main()
