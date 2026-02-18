#!/usr/bin/env python3
"""Split annotations into train and validation sets."""

import json
import random
from pathlib import Path

# Load annotations
annotations_path = Path('data/yearbook/annotations.json')
with open(annotations_path) as f:
    data = json.load(f)

# Shuffle and split
random.seed(42)
images = data['images']
random.shuffle(images)

split_idx = int(len(images) * 0.8)  # 80% train, 20% val
train_imgs = images[:split_idx]
val_imgs = images[split_idx:]

# Get image IDs
train_img_ids = {img['id'] for img in train_imgs}
val_img_ids = {img['id'] for img in val_imgs}

# Split annotations
train_annos = [a for a in data['annotations'] if a['image_id'] in train_img_ids]
val_annos = [a for a in data['annotations'] if a['image_id'] in val_img_ids]

# Create train/val datasets
train_data = {
    'images': train_imgs,
    'annotations': train_annos,
    'categories': data['categories']
}

val_data = {
    'images': val_imgs,
    'annotations': val_annos,
    'categories': data['categories']
}

# Save
with open('data/yearbook/annotations-train.json', 'w') as f:
    json.dump(train_data, f)

with open('data/yearbook/annotations-val.json', 'w') as f:
    json.dump(val_data, f)

print(f'✓ Train: {len(train_imgs)} images, {len(train_annos)} photos')
print(f'✓ Val: {len(val_imgs)} images, {len(val_annos)} photos')
print('\n✅ Data split complete!')
