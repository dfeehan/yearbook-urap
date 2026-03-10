#!/usr/bin/env python3
"""
Create a text-only dataset from the medium dataset.
Filters to include only text annotations (category_id=2).
"""

import json
from pathlib import Path

# Load the medium dataset
data_file = Path('layout-model-training/data/yearbook/annotations-train-medium.json')

with open(data_file) as f:
    data = json.load(f)

print(f"Original dataset:")
print(f"  Images: {len(data['images'])}")
print(f"  Annotations: {len(data['annotations'])}")

# Filter to only text annotations (category_id=2)
text_annotations = [ann for ann in data['annotations'] if ann['category_id'] == 2]

# Get image IDs that have text annotations
image_ids_with_text = set(ann['image_id'] for ann in text_annotations)

# Filter images to only those with text
text_images = [img for img in data['images'] if img['id'] in image_ids_with_text]

# Create text-only dataset
text_data = {
    "images": text_images,
    "annotations": text_annotations,
    "categories": [
        {"id": 1, "name": "text"}  # Renumber to category_id=1
    ]
}

# Renumber category_id from 2 to 1
for ann in text_data['annotations']:
    ann['category_id'] = 1

print(f"\nText-only dataset:")
print(f"  Images: {len(text_data['images'])}")
print(f"  Annotations: {len(text_data['annotations'])}")

# Save train set
output_file = Path('layout-model-training/data/yearbook/annotations-text-only-train.json')
output_file.parent.mkdir(parents=True, exist_ok=True)

with open(output_file, 'w') as f:
    json.dump(text_data, f, indent=2)

print(f"\n✓ Saved to {output_file}")

# Now do the same for validation set
val_file = Path('layout-model-training/data/yearbook/annotations-val-medium.json')

with open(val_file) as f:
    val_data = json.load(f)

print(f"\nOriginal validation:")
print(f"  Images: {len(val_data['images'])}")
print(f"  Annotations: {len(val_data['annotations'])}")

# Filter validation
val_text_annotations = [ann for ann in val_data['annotations'] if ann['category_id'] == 2]
val_image_ids_with_text = set(ann['image_id'] for ann in val_text_annotations)
val_text_images = [img for img in val_data['images'] if img['id'] in val_image_ids_with_text]

val_text_data = {
    "images": val_text_images,
    "annotations": val_text_annotations,
    "categories": [
        {"id": 1, "name": "text"}
    ]
}

# Renumber category_id
for ann in val_text_data['annotations']:
    ann['category_id'] = 1

print(f"\nText-only validation:")
print(f"  Images: {len(val_text_data['images'])}")
print(f"  Annotations: {len(val_text_data['annotations'])}")

val_output_file = Path('layout-model-training/data/yearbook/annotations-text-only-val.json')
with open(val_output_file, 'w') as f:
    json.dump(val_text_data, f, indent=2)

print(f"✓ Saved to {val_output_file}")
