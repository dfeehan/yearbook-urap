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

PHOTO_LABELS = {'Student Photo', 'Faculty Photo', 'Group Photo'}
TEXT_LABELS = {'Name', 'Additional Text'}
TEXT_CATEGORY_ID = 2
PHOTO_CATEGORY_ID = 1
DEDUP_IOU_THRESHOLD = 0.85
TEXT_PHOTO_OVERLAP_IOU = 0.5
MAX_TEXT_IMAGE_FRACTION = 0.05
MIN_TEXT_PORTRAIT_ASPECT = 0.35
MAX_TEXT_PORTRAIT_ASPECT = 1.3

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


def get_task_dimensions(task, results):
    data = task.get('data', {})
    width = data.get('width', 0)
    height = data.get('height', 0)

    if width == 0 or height == 0:
        for result in results:
            if 'original_width' in result and 'original_height' in result:
                width = result['original_width']
                height = result['original_height']
                break

    if width == 0 or height == 0:
        width = 2048
        height = 2048

    return width, height


def convert_result_to_box(result, fallback_width, fallback_height):
    value = result.get('value', {})
    labels = value.get('rectanglelabels', [])
    if not labels:
        return None

    label = labels[0]
    if label in PHOTO_LABELS:
        category_id = PHOTO_CATEGORY_ID
    elif label in TEXT_LABELS:
        category_id = TEXT_CATEGORY_ID
    else:
        return None

    img_w = result.get('original_width', fallback_width) or fallback_width
    img_h = result.get('original_height', fallback_height) or fallback_height

    x_pct = value.get('x', 0)
    y_pct = value.get('y', 0)
    w_pct = value.get('width', 0)
    h_pct = value.get('height', 0)

    x_abs = (x_pct / 100.0) * img_w
    y_abs = (y_pct / 100.0) * img_h
    w_abs = (w_pct / 100.0) * img_w
    h_abs = (h_pct / 100.0) * img_h

    if w_abs <= 0 or h_abs <= 0:
        return None

    return {
        'category_id': category_id,
        'label': label,
        'bbox': [x_abs, y_abs, w_abs, h_abs],
        'area': w_abs * h_abs,
    }


def bbox_iou(box_a, box_b):
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b

    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh

    ix1 = max(ax, bx)
    iy1 = max(ay, by)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0

    union = (aw * ah) + (bw * bh) - inter
    return inter / union if union > 0 else 0.0


def deduplicate_boxes(boxes):
    deduped = []
    for box in sorted(boxes, key=lambda entry: (entry['category_id'], -entry['area'])):
        duplicate = False
        for existing in deduped:
            if existing['category_id'] != box['category_id']:
                continue
            if bbox_iou(existing['bbox'], box['bbox']) >= DEDUP_IOU_THRESHOLD:
                duplicate = True
                break
        if not duplicate:
            deduped.append(box)
    return deduped


def is_suspicious_text_box(text_box, image_width, image_height, photo_boxes):
    x, y, width, height = text_box['bbox']
    if width <= 0 or height <= 0:
        return True

    image_area = image_width * image_height
    area_fraction = text_box['area'] / image_area if image_area > 0 else 0
    aspect_ratio = width / height

    is_large_portrait_like = (
        MIN_TEXT_PORTRAIT_ASPECT <= aspect_ratio <= MAX_TEXT_PORTRAIT_ASPECT
        and area_fraction >= MAX_TEXT_IMAGE_FRACTION
    )
    overlaps_photo = any(
        bbox_iou(text_box['bbox'], photo_box['bbox']) >= TEXT_PHOTO_OVERLAP_IOU
        for photo_box in photo_boxes
    )

    return is_large_portrait_like or overlaps_photo


def collect_useful_boxes(task):
    annotations = task.get('annotations', [])
    all_results = []
    for annotation in annotations:
        all_results.extend(annotation.get('result', []))

    if not all_results:
        return [], 2048, 2048

    width, height = get_task_dimensions(task, all_results)

    boxes = []
    for result in all_results:
        if result.get('type') != 'rectanglelabels':
            continue
        box = convert_result_to_box(result, width, height)
        if box is not None:
            boxes.append(box)

    boxes = deduplicate_boxes(boxes)

    photo_boxes = [box for box in boxes if box['category_id'] == PHOTO_CATEGORY_ID]
    filtered_boxes = []
    for box in boxes:
        if box['category_id'] == TEXT_CATEGORY_ID and is_suspicious_text_box(box, width, height, photo_boxes):
            continue
        filtered_boxes.append(box)

    return filtered_boxes, width, height

# Convert to COCO format
print("\nConverting to COCO format...")

coco_data = {
    "images": [],
    "annotations": [],
    "categories": [
        {"id": 1, "name": "human_figure"},  # Student Photo, Faculty Photo, Group Photo
        {"id": 2, "name": "text"}  # Name, Additional Text
    ]
}

annotation_id = 1
images_with_annotations = 0

for image_id, task in enumerate(tasks, start=1):
    boxes, width, height = collect_useful_boxes(task)

    if not boxes:
        continue

    # Get image info
    data = task.get('data', {})
    image_url = data.get('image', '')
    
    # Extract filename
    if '/' in image_url:
        filename = image_url.split('/')[-1]
    else:
        filename = f"task_{task['id']}.jpg"
    
    # Add image entry
    coco_data['images'].append({
        "id": image_id,
        "file_name": filename,
        "width": width if width > 0 else 2048,  # Default if unknown
        "height": height if height > 0 else 2048,
        "task_id": task['id']
    })
    
    has_valid_annotation = False

    for box in boxes:
        coco_data['annotations'].append({
            "id": annotation_id,
            "image_id": image_id,
            "category_id": box['category_id'],
            "bbox": box['bbox'],
            "area": box['area'],
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
