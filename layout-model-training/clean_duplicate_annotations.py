#!/usr/bin/env python3
"""
Clean duplicate annotations by removing text boxes that significantly overlap with human boxes
"""

import json
import numpy as np

def compute_iou(box1, box2):
    """Compute IoU between two boxes in [x, y, w, h] format"""
    x1_1, y1_1, w1, h1 = box1
    x1_2, y1_2, w2, h2 = box2
    
    x2_1, y2_1 = x1_1 + w1, y1_1 + h1
    x2_2, y2_2 = x1_2 + w2, y1_2 + h2
    
    xi1 = max(x1_1, x1_2)
    yi1 = max(y1_1, y1_2)
    xi2 = min(x2_1, x2_2)
    yi2 = min(y2_1, y2_2)
    
    if xi2 < xi1 or yi2 < yi1:
        return 0.0
    
    intersection = (xi2 - xi1) * (yi2 - yi1)
    area1 = w1 * h1
    area2 = w2 * h2
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0

# Load original annotations
with open('data/yearbook/annotations.json') as f:
    data = json.load(f)

print('=== Original annotations ===')
print(f'Total annotations: {len(data["annotations"])}')

human_anns = [a for a in data['annotations'] if a['category_id'] == 1]
text_anns = [a for a in data['annotations'] if a['category_id'] == 2]

print(f'Human (category 1): {len(human_anns)}')
print(f'Text (category 2): {len(text_anns)}')

# Find text boxes that overlap significantly with human boxes
removed = []
kept_text_anns = []

for text_ann in text_anns:
    text_bbox = text_ann['bbox']
    is_duplicate = False
    
    # Check overlap with all human boxes in the same image
    for human_ann in human_anns:
        if human_ann['image_id'] != text_ann['image_id']:
            continue
        
        human_bbox = human_ann['bbox']
        iou = compute_iou(text_bbox, human_bbox)
        
        # If text box overlaps >50% with a human box, it's likely a duplicate
        if iou > 0.5:
            is_duplicate = True
            removed.append({
                'text_ann_id': text_ann['id'],
                'human_ann_id': human_ann['id'],
                'iou': iou,
                'image_id': text_ann['image_id']
            })
            break
    
    if not is_duplicate:
        kept_text_anns.append(text_ann)

print(f'\n=== Cleaning results ===')
print(f'Removed {len(removed)} duplicate text annotations')
print(f'Kept {len(kept_text_anns)} legitimate text annotations')

# Create cleaned dataset
cleaned_annotations = human_anns + kept_text_anns

cleaned_data = {
    'images': data['images'],
    'annotations': cleaned_annotations,
    'categories': data['categories']
}

# Save cleaned version
output_file = 'data/yearbook/annotations-cleaned.json'
with open(output_file, 'w') as f:
    json.dump(cleaned_data, f, indent=2)

print(f'\n✅ Saved cleaned annotations to: {output_file}')
print(f'Total annotations: {len(cleaned_annotations)}')
print(f'  Human: {len(human_anns)}')
print(f'  Text: {len(kept_text_anns)}')
print(f'  Removed duplicates: {len(removed)}')
