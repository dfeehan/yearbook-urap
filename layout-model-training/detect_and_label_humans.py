#!/usr/bin/env python3
"""
Detect human figures in yearbook images and label them with numbers.
Uses the trained single-class human detection model.
"""

import cv2
import sys
import argparse
from pathlib import Path
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import numpy as np

def setup_cfg(model_path, config_file, threshold=0.5):
    """Setup configuration for the predictor."""
    cfg = get_cfg()
    cfg.merge_from_file(config_file)
    cfg.MODEL.WEIGHTS = model_path
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold
    cfg.MODEL.DEVICE = "cpu"
    cfg.freeze()
    return cfg

def detect_and_label(image_path, output_path, predictor, metadata):
    """Detect humans and label them with numbers."""
    # Read image
    im = cv2.imread(str(image_path))
    if im is None:
        print(f"❌ Could not read image: {image_path}")
        return
    
    # Run detection
    outputs = predictor(im)
    
    # Get predictions
    instances = outputs["instances"].to("cpu")
    boxes = instances.pred_boxes.tensor.numpy()
    scores = instances.scores.numpy()
    
    print(f"\n📷 Image: {image_path.name}")
    print(f"   Detected {len(boxes)} human figures")
    
    # Sort by Y-coordinate (top to bottom), then X-coordinate (left to right)
    # This gives a natural reading order
    centers = [(box[0] + box[2]) / 2 for box in boxes]  # X centers
    y_coords = [(box[1] + box[3]) / 2 for box in boxes]  # Y centers
    
    # Sort by Y first (with some tolerance for same row), then by X
    indices = sorted(range(len(boxes)), key=lambda i: (y_coords[i] // 50, centers[i]))
    
    # Draw boxes and labels
    for rank, idx in enumerate(indices, 1):
        box = boxes[idx]
        score = scores[idx]
        
        x1, y1, x2, y2 = box.astype(int)
        
        # Draw bounding box
        cv2.rectangle(im, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Create label with number and confidence
        label = f"{rank} ({score:.2f})"
        
        # Draw label background
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        thickness = 2
        (label_width, label_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        
        # Position label above box, or inside if box is at top
        if y1 - label_height - 10 > 0:
            label_y = y1 - 10
            bg_y1, bg_y2 = y1 - label_height - 15, y1 - 5
        else:
            label_y = y1 + label_height + 10
            bg_y1, bg_y2 = y1 + 5, y1 + label_height + 15
        
        # Draw white background for label
        cv2.rectangle(im, (x1, bg_y1), (x1 + label_width + 10, bg_y2), (255, 255, 255), -1)
        
        # Draw label text
        cv2.putText(im, label, (x1 + 5, label_y), font, font_scale, (0, 0, 0), thickness)
        
        print(f"   #{rank}: Box at ({x1},{y1}) - ({x2},{y2}), confidence: {score:.2%}")
    
    # Save output
    cv2.imwrite(str(output_path), im)
    print(f"✓ Saved labeled image to: {output_path}")
    
    return len(boxes)

def main():
    parser = argparse.ArgumentParser(description="Detect and label human figures in yearbook images")
    parser.add_argument("image", help="Path to input image")
    parser.add_argument("--model", default="outputs/yearbook/fast_rcnn_R_50_FPN_3x/model_final.pth",
                        help="Path to trained model weights")
    parser.add_argument("--config", default="configs/yearbook/fast_rcnn_R_50_FPN_3x.yaml",
                        help="Path to config file")
    parser.add_argument("--output", help="Path to output image (default: same dir with _labeled suffix)")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Detection confidence threshold (default: 0.5)")
    
    args = parser.parse_args()
    
    # Resolve paths
    image_path = Path(args.image)
    if not image_path.exists():
        print(f"❌ Image not found: {image_path}")
        sys.exit(1)
    
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        sys.exit(1)
    
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"❌ Config not found: {config_path}")
        sys.exit(1)
    
    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = image_path.parent / f"{image_path.stem}_labeled{image_path.suffix}"
    
    # Setup predictor
    cfg = setup_cfg(str(model_path), str(config_path), args.threshold)
    predictor = DefaultPredictor(cfg)
    
    # Register metadata
    metadata = MetadataCatalog.get("yearbook_inference")
    metadata.thing_classes = ["human"]
    
    # Run detection
    detect_and_label(image_path, output_path, predictor, metadata)

if __name__ == "__main__":
    main()
