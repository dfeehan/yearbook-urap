#!/usr/bin/env python3
import json
import random
from collections import Counter
from pathlib import Path

# Load full annotations
with open('layout-model-training/data/yearbook/annotations.json') as f:
    data = json.load(f)

# Filter to only images that exist on disk
images_dir = Path('layout-model-training/data/yearbook/images')
existing_images = []
for img in data['images']:
    img_path = images_dir / img['file_name']
    if img_path.exists():
        existing_images.append(img)

print(f"Total images in annotations: {len(data['images'])}")
print(f"Images actually downloaded: {len(existing_images)}")

# Use all available images, split 80/20
random.seed(42)
random.shuffle(existing_images)

train_count = int(len(existing_images) * 0.8)
train_images = existing_images[:train_count]
val_images = existing_images[train_count:]

train_image_ids = {img['id'] for img in train_images}
val_image_ids = {img['id'] for img in val_images}

# Split annotations
train_annotations = [ann for ann in data['annotations'] if ann['image_id'] in train_image_ids]
val_annotations = [ann for ann in data['annotations'] if ann['image_id'] in val_image_ids]

# Create train set
train_data = {
    'images': train_images,
    'annotations': train_annotations,
    'categories': data['categories']
}

# Create val set
val_data = {
    'images': val_images,
    'annotations': val_annotations,
    'categories': data['categories']
}

# Save
with open('layout-model-training/data/yearbook/annotations-train-medium.json', 'w') as f:
    json.dump(train_data, f)

with open('layout-model-training/data/yearbook/annotations-val-medium.json', 'w') as f:
    json.dump(val_data, f)

train_counts = Counter(ann['category_id'] for ann in train_annotations)
val_counts = Counter(ann['category_id'] for ann in val_annotations)

print(f"\n✅ Medium dataset created using available images:")
print(f"   Train: {len(train_images)} images, {len(train_annotations)} annotations")
print(f"     - Human figures: {train_counts.get(1, 0)}")
print(f"     - Text boxes: {train_counts.get(2, 0)}")
print(f"   Val: {len(val_images)} images, {len(val_annotations)} annotations")
print(f"     - Human figures: {val_counts.get(1, 0)}")
print(f"     - Text boxes: {val_counts.get(2, 0)}")
