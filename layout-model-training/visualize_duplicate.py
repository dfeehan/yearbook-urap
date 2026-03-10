#!/usr/bin/env python3
"""
Visualize both the human and text boxes for the problematic annotation
"""

import json
import cv2
from pathlib import Path

# Load annotations
with open('data/yearbook/annotations-train-medium.json') as f:
    data = json.load(f)

# Find the image
target_image = 'ayantee1939negr_pages_22_23.jpg'
img_obj = next((img for img in data['images'] if img['file_name'] == target_image), None)
img_id = img_obj['id']

# Get the duplicate box annotations
target_bbox = [1301.9375672766414, 1037.2284345047922, 264.800861141012, 357.51277955271576]

# Load image
img_path = Path('data/yearbook/images') / target_image
im = cv2.imread(str(img_path))

# Get all annotations for this image
image_anns = [ann for ann in data['annotations'] if ann['image_id'] == img_id]

# Draw category 1 (human) boxes in blue
human_anns = [ann for ann in image_anns if ann['category_id'] == 1]
for ann in human_anns:
    x, y, w, h = ann['bbox']
    cv2.rectangle(im, (int(x), int(y)), (int(x+w), int(y+h)), (255, 0, 0), 3)
    cv2.putText(im, 'HUMAN', (int(x), int(y)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

# Draw category 2 (text) boxes in green
text_anns = [ann for ann in image_anns if ann['category_id'] == 2]
for ann in text_anns:
    x, y, w, h = ann['bbox']
    cv2.rectangle(im, (int(x), int(y)), (int(x+w), int(y+h)), (0, 255, 0), 2)
    cv2.putText(im, f'TEXT {w:.0f}x{h:.0f}', (int(x), int(y)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

# Highlight the problematic box
x, y, w, h = target_bbox
cv2.rectangle(im, (int(x), int(y)), (int(x+w), int(y+h)), (0, 0, 255), 5)
cv2.putText(im, 'DUPLICATE BOX', (int(x), int(y)+30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

output_path = 'outputs/duplicate_annotation_visualization.jpg'
cv2.imwrite(output_path, im)
print(f'Saved visualization showing all boxes')
print(f'Blue = Human annotations')
print(f'Green = Text annotations')
print(f'Red thick = The duplicate 265x358 box')
print(f'Output: {output_path}')

print(f'\n=== Summary ===')
print(f'Total annotations on this page: {len(image_anns)}')
print(f'Human (category 1): {len(human_anns)}')
print(f'Text (category 2): {len(text_anns)}')
