#!/usr/bin/env python3
"""
Fetch annotations directly from Label Studio tasks API and convert to COCO format.
This works even without export permissions.
"""

import os
import sys
import requests
import json
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

API_TOKEN = os.getenv('LABEL_STUDIO_TOKEN')
BASE_URL = os.getenv('LABEL_STUDIO_BASE_URL', 'https://app.humansignal.com')

if not API_TOKEN:
    print("❌ Error: LABEL_STUDIO_TOKEN not found")
    sys.exit(1)

PROJECT_ID = 158111  # Your yearbook project

print("=" * 70)
print("Fetching Annotations from Label Studio Tasks")
print("=" * 70)

# Fetch all tasks with pagination
print(f"\nFetching tasks from project {PROJECT_ID}...")

all_tasks = []
page = 1
page_size = 100

while True:
    response = requests.get(
        f"{BASE_URL}/api/projects/{PROJECT_ID}/tasks",
        params={'page': page, 'page_size': page_size},
        headers={"Authorization": f"Token {API_TOKEN}"}
    )
    
    if response.status_code != 200:
        print(f"❌ Failed: HTTP {response.status_code}")
        print(f"Response: {response.text}")
        sys.exit(1)
    
    data = response.json()
    
    # Handle different response formats
    if isinstance(data, list):
        tasks_page = data
    elif isinstance(data, dict):
        tasks_page = data.get('tasks', data.get('results', []))
    else:
        break
    
    if not tasks_page:
        break
    
    all_tasks.extend(tasks_page)
    print(f"  Page {page}: {len(tasks_page)} tasks")
    
    # Check if there are more pages
    if isinstance(data, dict) and data.get('next'):
        page += 1
    elif len(tasks_page) < page_size:
        break
    else:
        page += 1

tasks = all_tasks
print(f"✓ Found {len(tasks)} total tasks")

# Convert to COCO format
print("\nConverting to COCO format...")

coco_data = {
    "images": [],
    "annotations": [],
    "categories": [{"id": 1, "name": "human_figure"}]  # Student, Faculty, and Group photos
}

annotation_id = 1
images_with_annotations = 0

for image_id, task in enumerate(tasks, start=1):
    # Get annotations
    annotations = task.get('annotations', [])
    if not annotations:
        continue
    
    # Use the first (usually only) annotation per task
    annotation = annotations[0]
    results = annotation.get('result', [])
    
    if not results:
        continue
    
    # Get image info
    data = task.get('data', {})
    image_url = data.get('image', '')
    
    # Extract filename
    if '/' in image_url:
        filename = image_url.split('/')[-1]
    else:
        filename = f"task_{task['id']}.jpg"
    
    # Get image dimensions (often stored in the task data)
    width = data.get('width', 0)
    height = data.get('height', 0)
    
    # If dimensions not in data, try to get from annotation
    if width == 0 or height == 0:
        for result in results:
            if 'original_width' in result:
                width = result['original_width']
                height = result['original_height']
                break
    
    # Add image entry
    coco_data['images'].append({
        "id": image_id,
        "file_name": filename,
        "width": width if width > 0 else 2048,  # Default if unknown
        "height": height if height > 0 else 2048,
        "task_id": task['id']
    })
    
    has_valid_annotation = False
    
    # Process each annotation result
    for result in results:
        if result.get('type') != 'rectanglelabels':
            continue
        
        value = result.get('value', {})
        
        # Only include photo labels (human figures), skip text labels
        labels = value.get('rectanglelabels', [])
        if not labels:
            continue
        
        label = labels[0]
        if label not in ['Student Photo', 'Faculty Photo', 'Group Photo']:
            # Skip text boxes like 'Name' and 'Additional Text'
            continue
        
        # Get bounding box (stored as percentages)
        x_pct = value.get('x', 0)
        y_pct = value.get('y', 0)
        w_pct = value.get('width', 0)
        h_pct = value.get('height', 0)
        
        # Get actual dimensions
        img_w = result.get('original_width', width)
        img_h = result.get('original_height', height)
        
        if img_w == 0 or img_h == 0:
            # Use default large dimensions if unknown
            img_w = 2048
            img_h = 2048
        
        # Convert percentages to absolute coordinates
        x_abs = (x_pct / 100.0) * img_w
        y_abs = (y_pct / 100.0) * img_h
        w_abs = (w_pct / 100.0) * img_w
        h_abs = (h_pct / 100.0) * img_h
        
        # COCO format: [x, y, width, height]
        bbox = [x_abs, y_abs, w_abs, h_abs]
        area = w_abs * h_abs
        
        coco_data['annotations'].append({
            "id": annotation_id,
            "image_id": image_id,
            "category_id": 1,
            "bbox": bbox,
            "area": area,
            "iscrowd": 0
        })
        annotation_id += 1
        has_valid_annotation = True
    
    if has_valid_annotation:
        images_with_annotations += 1

print(f"✓ Processed {images_with_annotations} images with annotations")
print(f"✓ Total bounding boxes: {len(coco_data['annotations'])}")

if images_with_annotations == 0:
    print("\n❌ No annotated images found!")
    print("Make sure your tasks have completed annotations.")
    sys.exit(1)

# Save to file
output_path = Path('layout-model-training/data/yearbook/annotations.json')
output_path.parent.mkdir(parents=True, exist_ok=True)

with open(output_path, 'w') as f:
    json.dump(coco_data, f, indent=2)

print(f"\n✅ Saved to: {output_path}")
print("\n" + "=" * 70)
print("Statistics:")
print("=" * 70)
print(f"  Images: {len(coco_data['images'])}")
print(f"  Annotations: {len(coco_data['annotations'])}")
print(f"  Avg photos per page: {len(coco_data['annotations']) / len(coco_data['images']):.1f}")
print(f"  Categories: {[c['name'] for c in coco_data['categories']]}")
print("=" * 70)
print("\n✅ Ready for training!")
print("\nNext steps:")
print("  1. Split data: see EXPORT_ANNOTATIONS.md")
print("  2. Train: cd layout-model-training/scripts && ./train_yearbook.sh")
