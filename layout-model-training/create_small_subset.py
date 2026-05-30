#!/usr/bin/env python3
"""Create a small training subset for quick testing."""

import json
import random
from pathlib import Path

# Load full annotations
with open('layout-model-training/data/yearbook/annotations.json') as f:
    data = json.load(f)

# Select first 50 images (or random subset)
random.seed(42)
all_images = data['images']
subset_images = all_images[:50]  # First 50 images

subset_img_ids = {img['id'] for img in subset_images}

# Filter annotations for subset
subset_annos = [a for a in data['annotations'] if a['image_id'] in subset_img_ids]

# Create subset data
subset_data = {
    'images': subset_images,
    'annotations': subset_annos,
    'categories': data['categories']
}

# Split into train (40) / val (10)
train_imgs = subset_images[:40]
val_imgs = subset_images[40:50]

train_img_ids = {img['id'] for img in train_imgs}
val_img_ids = {img['id'] for img in val_imgs}

train_annos = [a for a in subset_annos if a['image_id'] in train_img_ids]
val_annos = [a for a in subset_annos if a['image_id'] in val_img_ids]

train_data = {'images': train_imgs, 'annotations': train_annos, 'categories': data['categories']}
val_data = {'images': val_imgs, 'annotations': val_annos, 'categories': data['categories']}

# Save
Path('layout-model-training/data/yearbook').mkdir(parents=True, exist_ok=True)

with open('layout-model-training/data/yearbook/annotations-small.json', 'w') as f:
    json.dump(subset_data, f)

with open('layout-model-training/data/yearbook/annotations-train-small.json', 'w') as f:
    json.dump(train_data, f)

with open('layout-model-training/data/yearbook/annotations-val-small.json', 'w') as f:
    json.dump(val_data, f)

print(f'✅ Small subset created:')
print(f'   Train: {len(train_imgs)} images, {len(train_annos)} photos')
print(f'   Val: {len(val_imgs)} images, {len(val_annos)} photos')
print(f'   Total annotations: {len(subset_annos)}')
print(f'\n📋 Image list for download:')

# Save list of images to download
with open('layout-model-training/data/yearbook/images_to_download.txt', 'w') as f:
    for img in subset_images:
        f.write(f"{img['file_name']}\n")
        print(f"  {img['file_name']}")
