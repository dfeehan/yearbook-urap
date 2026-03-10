#!/usr/bin/env python3
"""
Visualize large text boxes to see if they're actually portraits
"""

import json
import cv2
import numpy as np
from pathlib import Path

# Load annotations
with open('data/yearbook/annotations-train-medium.json') as f:
    data = json.load(f)

# Find large text boxes (category_id=2, size >100x100)
large_text_boxes = []
for ann in data['annotations']:
    if ann['category_id'] == 2:  # text category
        w, h = ann['bbox'][2], ann['bbox'][3]
        if w > 100 and h > 100:
            img_id = ann['image_id']
            img = next(img for img in data['images'] if img['id'] == img_id)
            large_text_boxes.append({
                'image': img['file_name'],
                'bbox': ann['bbox'],
                'size': f'{w:.0f}x{h:.0f}'
            })

print(f'Found {len(large_text_boxes)} large "text" boxes (>100x100)')
print('\nVisualizing first 3 examples...')

# Visualize first 3
for i, box_info in enumerate(large_text_boxes[:3]):
    img_path = Path('data/yearbook/images') / box_info['image']
    if not img_path.exists():
        print(f'Image not found: {img_path}')
        continue
    
    im = cv2.imread(str(img_path))
    bbox = box_info['bbox']  # [x, y, width, height]
    x, y, w, h = bbox
    
    # Draw box
    cv2.rectangle(im, (int(x), int(y)), (int(x+w), int(y+h)), (0, 255, 0), 3)
    
    # Add label
    label = f'Text box: {box_info["size"]}'
    cv2.putText(im, label, (int(x), int(y)-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    output_path = f'outputs/large_text_example_{i+1}.jpg'
    cv2.imwrite(output_path, im)
    print(f'  Example {i+1}: {box_info["size"]} in {box_info["image"]}')
    print(f'    Saved to: {output_path}')
