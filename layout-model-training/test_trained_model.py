#!/usr/bin/env python3
"""
Test script for trained yearbook layout detection model.

Usage:
    python test_trained_model.py --input yearbook_page.jpg --output results/
"""

import layoutparser as lp
import cv2
import matplotlib.pyplot as plt
import argparse
from pathlib import Path

def test_model(image_path, model_path, config_path, output_dir, confidence=0.5):
    """Test the trained model on a yearbook page."""
    
    # Load the trained model
    print(f"Loading model from: {model_path}")
    model = lp.Detectron2LayoutModel(
        config_path=config_path,
        model_path=model_path,
        extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", confidence],
        label_map={0: "photo"}  # Adjust if you have multiple classes
    )
    
    # Load and process image
    print(f"Processing: {image_path}")
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Detect layout
    layout = model.detect(image)
    print(f"✓ Detected {len(layout)} photos")
    
    # Draw results
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    vis_image = lp.draw_box(image_rgb, layout, box_width=5)
    
    # Save visualization
    output_path = Path(output_dir) / f"{Path(image_path).stem}_detected.jpg"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert back to BGR for saving
    import numpy as np
    if isinstance(vis_image, np.ndarray):
        vis_image_bgr = cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR)
    else:
        # Handle PIL Image
        vis_image_bgr = cv2.cvtColor(np.array(vis_image), cv2.COLOR_RGB2BGR)
    
    cv2.imwrite(str(output_path), vis_image_bgr)
    print(f"✓ Saved visualization to: {output_path}")
    
    # Print detection details
    print("\nDetection Details:")
    for i, block in enumerate(layout):
        x1, y1, x2, y2 = block.coordinates
        w, h = x2 - x1, y2 - y1
        conf = block.score
        print(f"  Photo {i+1}: bbox=({x1:.0f}, {y1:.0f}, {w:.0f}x{h:.0f}), confidence={conf:.3f}")
    
    return layout

def main():
    parser = argparse.ArgumentParser(description='Test trained yearbook layout model')
    parser.add_argument('--input', required=True, help='Input yearbook page image')
    parser.add_argument('--output', default='test_results/', help='Output directory')
    parser.add_argument('--model', default='outputs/yearbook/fast_rcnn_R_50_FPN_3x/model_final.pth',
                       help='Path to trained model weights')
    parser.add_argument('--config', default='configs/yearbook/fast_rcnn_R_50_FPN_3x.yaml',
                       help='Path to model config file')
    parser.add_argument('--confidence', type=float, default=0.5,
                       help='Detection confidence threshold (0-1)')
    
    args = parser.parse_args()
    
    # Get absolute paths
    base_dir = Path(__file__).parent
    model_path = base_dir / args.model
    config_path = base_dir / args.config
    
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        print("Have you trained the model yet? See TRAINING_GUIDE.md")
        return
    
    if not config_path.exists():
        print(f"❌ Config not found: {config_path}")
        return
    
    # Test the model
    layout = test_model(
        image_path=args.input,
        model_path=str(model_path),
        config_path=str(config_path),
        output_dir=args.output,
        confidence=args.confidence
    )
    
    print(f"\n✅ Testing complete!")
    print(f"Found {len(layout)} student photos with confidence >= {args.confidence}")

if __name__ == "__main__":
    main()
