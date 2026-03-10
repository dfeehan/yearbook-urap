#!/usr/bin/env python3
"""
Debug the specific 265x358 box to see why it's categorized as text
"""

import json

# Load annotations
with open('data/yearbook/annotations-train-medium.json') as f:
    data = json.load(f)

# Find the specific box: 265x358 in ayantee1939negr_pages_22_23.jpg
target_image = 'ayantee1939negr_pages_22_23.jpg'
target_w, target_h = 265, 358

# Get image ID
img = next((img for img in data['images'] if img['file_name'] == target_image), None)
if not img:
    print(f'Image not found: {target_image}')
    exit(1)

img_id = img['id']
print(f'Image: {target_image}')
print(f'Image ID: {img_id}')

# Find all annotations for this image
image_annotations = [ann for ann in data['annotations'] if ann['image_id'] == img_id]
print(f'\nTotal annotations for this image: {len(image_annotations)}')

# Look for boxes close to 265x358
print(f'\nSearching for box ~{target_w}x{target_h}:')
for ann in image_annotations:
    bbox = ann['bbox']
    w, h = bbox[2], bbox[3]
    if abs(w - target_w) < 5 and abs(h - target_h) < 5:  # Close match
        cat_id = ann['category_id']
        cat_name = next(c['name'] for c in data['categories'] if c['id'] == cat_id)
        print(f'\n  Found matching box:')
        print(f'    BBox: {bbox}')
        print(f'    Size: {w:.1f}x{h:.1f}')
        print(f'    Category ID: {cat_id}')
        print(f'    Category Name: {cat_name}')
        print(f'    Annotation ID: {ann["id"]}')

# Show all boxes in this image by category
cat1 = [ann for ann in image_annotations if ann['category_id'] == 1]
cat2 = [ann for ann in image_annotations if ann['category_id'] == 2]

print(f'\n=== All annotations in this image ===')
print(f'Category 1 (human_figure): {len(cat1)} boxes')
print(f'Category 2 (text): {len(cat2)} boxes')

# Check if there are any overlapping boxes with different categories
print(f'\nCategory 2 (text) boxes in this image:')
for i, ann in enumerate(cat2[:10]):
    bbox = ann['bbox']
    print(f'  {i+1}. Size: {bbox[2]:.0f}x{bbox[3]:.0f}, Position: ({bbox[0]:.0f}, {bbox[1]:.0f})')
