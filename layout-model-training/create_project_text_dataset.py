#!/usr/bin/env python3
"""Build a clean text-only COCO dataset from all annotated tasks in a Label Studio project.

This exporter is designed for the new clean text annotation project. It keeps only
the latest non-empty annotation per task, removes near-duplicate boxes, writes a
group-diverse train/val split, and saves a CSV report of suspicious heavy overlaps
for manual review.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import requests
from dotenv import load_dotenv

TEXT_LABELS = {"Text", "Name", "Additional Text"}
TEXT_CATEGORY_ID = 1
DEFAULT_BASE_URL = "https://app.humansignal.com"
DEFAULT_IMAGES_DIR = "layout-model-training/data/yearbook/images"
DEFAULT_OUTPUT_TRAIN = "layout-model-training/data/yearbook/annotations-clean60-train.json"
DEFAULT_OUTPUT_VAL = "layout-model-training/data/yearbook/annotations-clean60-val.json"
DEFAULT_OVERLAP_REPORT = "layout-model-training/data/yearbook/annotations-clean60-overlap-review.csv"
DEFAULT_DEDUP_IOU = 0.85
DEFAULT_REVIEW_IOU = 0.35
DEFAULT_REVIEW_CONTAINMENT = 0.70
DEFAULT_VAL_RATIO = 0.20
DEFAULT_FALLBACK_DIMENSION = 2048


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-id", type=int, required=True, help="Label Studio project ID")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="Label Studio base URL")
    parser.add_argument("--images-dir", default=DEFAULT_IMAGES_DIR, help="Local yearbook image directory")
    parser.add_argument("--output-train", default=DEFAULT_OUTPUT_TRAIN, help="Train COCO JSON output path")
    parser.add_argument("--output-val", default=DEFAULT_OUTPUT_VAL, help="Validation COCO JSON output path")
    parser.add_argument(
        "--overlap-report",
        default=DEFAULT_OVERLAP_REPORT,
        help="CSV report path for suspicious remaining box overlaps",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for the train/val split")
    parser.add_argument("--val-ratio", type=float, default=DEFAULT_VAL_RATIO, help="Target validation fraction")
    parser.add_argument(
        "--dedup-iou-threshold",
        type=float,
        default=DEFAULT_DEDUP_IOU,
        help="Drop boxes with IoU above this threshold as duplicates",
    )
    parser.add_argument(
        "--review-iou-threshold",
        type=float,
        default=DEFAULT_REVIEW_IOU,
        help="Report remaining overlaps with IoU above this threshold",
    )
    parser.add_argument(
        "--review-containment-threshold",
        type=float,
        default=DEFAULT_REVIEW_CONTAINMENT,
        help="Report remaining overlaps when the intersection covers this much of the smaller box",
    )
    parser.add_argument(
        "--download-missing-images",
        action="store_true",
        help="Download missing images through Label Studio before writing COCO JSON files",
    )
    parser.add_argument(
        "--allow-missing-images",
        action="store_true",
        help="Do not fail when a referenced image is not present locally",
    )
    return parser.parse_args()


def build_session(base_url: str) -> tuple[requests.Session, str]:
    load_dotenv(dotenv_path=".env")
    token = os.getenv("LABEL_STUDIO_TOKEN")
    if not token:
        raise RuntimeError("LABEL_STUDIO_TOKEN not found in environment or .env")

    session = requests.Session()
    session.headers.update({"Authorization": f"Token {token}"})
    return session, base_url.rstrip("/")


def fetch_project_tasks(session: requests.Session, base_url: str, project_id: int) -> list[dict[str, Any]]:
    all_tasks: list[dict[str, Any]] = []
    page = 1
    page_size = 100

    while True:
        response = session.get(
            f"{base_url}/api/projects/{project_id}/tasks",
            params={"page": page, "page_size": page_size},
            timeout=60,
        )
        response.raise_for_status()
        data = response.json()

        if isinstance(data, list):
            page_tasks = data
        elif isinstance(data, dict):
            page_tasks = data.get("tasks", data.get("results", []))
        else:
            page_tasks = []

        if not page_tasks:
            break

        all_tasks.extend(page_tasks)

        if isinstance(data, dict) and data.get("next"):
            page += 1
        elif len(page_tasks) < page_size:
            break
        else:
            page += 1

    return all_tasks


def normalize_uploaded_filename(filename: str) -> str:
    return re.sub(r"^[0-9a-f]{8}-", "", filename)


def extract_image_url(task: dict[str, Any]) -> str:
    return task.get("data", {}).get("image", "")


def extract_filename(task: dict[str, Any]) -> str:
    image_url = extract_image_url(task)
    if not image_url:
        return f"task_{task['id']}.jpg"

    parsed = urlparse(image_url)
    query = parse_qs(parsed.query)
    if "fileuri" in query:
        value = query["fileuri"][0]
        return normalize_uploaded_filename(value.rsplit("/", 1)[-1])

    path = parsed.path or image_url
    name = path.rsplit("/", 1)[-1]
    return normalize_uploaded_filename(name or f"task_{task['id']}.jpg")


def source_group_from_filename(filename: str) -> str:
    return filename.split("_pages_", 1)[0]


def rectangle_results(annotation: dict[str, Any]) -> list[dict[str, Any]]:
    return [result for result in annotation.get("result", []) if result.get("type") == "rectanglelabels"]


def select_latest_nonempty_annotation(task: dict[str, Any]) -> dict[str, Any] | None:
    candidates = []
    for annotation in task.get("annotations", []):
        results = rectangle_results(annotation)
        if results:
            candidates.append(annotation)

    if not candidates:
        return None

    return max(
        candidates,
        key=lambda annotation: (
            annotation.get("updated_at") or annotation.get("created_at") or "",
            annotation.get("id") or 0,
        ),
    )


def get_task_dimensions(task: dict[str, Any], results: list[dict[str, Any]]) -> tuple[int, int]:
    data = task.get("data", {})
    width = data.get("width", 0)
    height = data.get("height", 0)

    if not width or not height:
        for result in results:
            if result.get("original_width") and result.get("original_height"):
                width = result["original_width"]
                height = result["original_height"]
                break

    if not width or not height:
        width = DEFAULT_FALLBACK_DIMENSION
        height = DEFAULT_FALLBACK_DIMENSION

    return int(width), int(height)


def bbox_iou(box_a: list[float], box_b: list[float]) -> float:
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh

    ix1 = max(ax, bx)
    iy1 = max(ay, by)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def containment_fraction(box_a: list[float], box_b: list[float]) -> float:
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh

    ix1 = max(ax, bx)
    iy1 = max(ay, by)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    smaller = min(aw * ah, bw * bh)
    return inter / smaller if smaller > 0 else 0.0


def convert_text_boxes(
    task: dict[str, Any],
    annotation: dict[str, Any],
    dedup_iou_threshold: float,
) -> tuple[list[dict[str, Any]], int, int, int]:
    results = rectangle_results(annotation)
    width, height = get_task_dimensions(task, results)
    boxes = []

    for result in results:
        value = result.get("value", {})
        labels = value.get("rectanglelabels", [])
        if not labels or labels[0] not in TEXT_LABELS:
            continue

        img_w = float(result.get("original_width", width) or width)
        img_h = float(result.get("original_height", height) or height)
        x_abs = (float(value.get("x", 0)) / 100.0) * img_w
        y_abs = (float(value.get("y", 0)) / 100.0) * img_h
        w_abs = (float(value.get("width", 0)) / 100.0) * img_w
        h_abs = (float(value.get("height", 0)) / 100.0) * img_h

        if w_abs <= 0 or h_abs <= 0:
            continue

        boxes.append({
            "bbox": [x_abs, y_abs, w_abs, h_abs],
            "area": w_abs * h_abs,
        })

    deduped = []
    removed_duplicates = 0
    for box in sorted(boxes, key=lambda entry: -entry["area"]):
        duplicate = False
        for existing in deduped:
            if bbox_iou(existing["bbox"], box["bbox"]) >= dedup_iou_threshold:
                duplicate = True
                removed_duplicates += 1
                break
        if not duplicate:
            deduped.append(box)

    return deduped, width, height, removed_duplicates


def build_overlap_rows(
    task_id: int,
    filename: str,
    boxes: list[dict[str, Any]],
    review_iou_threshold: float,
    review_containment_threshold: float,
) -> list[dict[str, Any]]:
    rows = []
    for index_a in range(len(boxes)):
        for index_b in range(index_a + 1, len(boxes)):
            box_a = boxes[index_a]["bbox"]
            box_b = boxes[index_b]["bbox"]
            iou_value = bbox_iou(box_a, box_b)
            contain_value = containment_fraction(box_a, box_b)
            if iou_value < review_iou_threshold and contain_value < review_containment_threshold:
                continue
            rows.append(
                {
                    "task_id": task_id,
                    "file_name": filename,
                    "box_a_index": index_a,
                    "box_b_index": index_b,
                    "iou": round(iou_value, 4),
                    "containment": round(contain_value, 4),
                    "box_a": json.dumps([round(value, 2) for value in box_a]),
                    "box_b": json.dumps([round(value, 2) for value in box_b]),
                }
            )
    return rows


def download_image(session: requests.Session, base_url: str, image_url: str, output_path: Path) -> None:
    if image_url.startswith("http://") or image_url.startswith("https://"):
        full_url = image_url
    else:
        full_url = f"{base_url}/{image_url.lstrip('/')}"

    response = session.get(full_url, timeout=120)
    response.raise_for_status()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(response.content)


def validate_and_download_images(
    session: requests.Session,
    base_url: str,
    task_records: list[dict[str, Any]],
    images_dir: Path,
    allow_missing: bool,
    download: bool,
) -> list[str]:
    missing = []
    for record in task_records:
        output_path = images_dir / record["file_name"]
        if output_path.exists():
            continue
        if download:
            try:
                download_image(session, base_url, record["image_url"], output_path)
                continue
            except Exception:
                pass
        missing.append(record["file_name"])

    if missing and not allow_missing:
        preview = ", ".join(missing[:10])
        raise RuntimeError(
            f"Missing {len(missing)} images in {images_dir}: {preview}{' ...' if len(missing) > 10 else ''}"
        )
    return missing


def build_task_records(
    tasks: list[dict[str, Any]],
    dedup_iou_threshold: float,
    review_iou_threshold: float,
    review_containment_threshold: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    records = []
    overlap_rows = []
    stats = {
        "annotated_tasks": 0,
        "skipped_empty_tasks": 0,
        "dedup_removed": 0,
    }

    for task in tasks:
        annotation = select_latest_nonempty_annotation(task)
        if annotation is None:
            stats["skipped_empty_tasks"] += 1
            continue

        filename = extract_filename(task)
        boxes, width, height, removed_duplicates = convert_text_boxes(task, annotation, dedup_iou_threshold)
        if not boxes:
            stats["skipped_empty_tasks"] += 1
            continue

        stats["annotated_tasks"] += 1
        stats["dedup_removed"] += removed_duplicates

        record = {
            "task_id": task["id"],
            "annotation_id": annotation.get("id"),
            "file_name": filename,
            "width": width,
            "height": height,
            "image_url": extract_image_url(task),
            "source_group": source_group_from_filename(filename),
            "boxes": boxes,
        }
        records.append(record)
        overlap_rows.extend(
            build_overlap_rows(
                task["id"],
                filename,
                boxes,
                review_iou_threshold,
                review_containment_threshold,
            )
        )

    return records, overlap_rows, stats


def split_records(records: list[dict[str, Any]], val_ratio: float, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not records:
        return [], []

    rng = random.Random(seed)
    target_val_count = max(1, round(len(records) * val_ratio))
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[record["source_group"]].append(record)
    for group_records in groups.values():
        rng.shuffle(group_records)

    group_names = list(groups.keys())
    rng.shuffle(group_names)

    val_records = []
    used_task_ids = set()

    # First pass: diversify the validation set across yearbook sources while
    # keeping at least one page from that source in train when possible.
    for group_name in group_names:
        if len(val_records) >= target_val_count:
            break
        group_records = groups[group_name]
        if len(group_records) < 2:
            continue
        candidate = group_records.pop()
        val_records.append(candidate)
        used_task_ids.add(candidate["task_id"])

    remaining = [record for record in records if record["task_id"] not in used_task_ids]
    rng.shuffle(remaining)
    while len(val_records) < target_val_count and remaining:
        candidate = remaining.pop()
        val_records.append(candidate)
        used_task_ids.add(candidate["task_id"])

    train_records = [record for record in records if record["task_id"] not in used_task_ids]
    return train_records, val_records


def build_coco(records: list[dict[str, Any]]) -> dict[str, Any]:
    images = []
    annotations = []
    annotation_id = 1

    for image_id, record in enumerate(records, start=1):
        images.append(
            {
                "id": image_id,
                "file_name": record["file_name"],
                "width": record["width"],
                "height": record["height"],
                "task_id": record["task_id"],
                "annotation_id": record["annotation_id"],
                "source_group": record["source_group"],
            }
        )
        for box in record["boxes"]:
            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": TEXT_CATEGORY_ID,
                    "bbox": box["bbox"],
                    "area": box["area"],
                    "iscrowd": 0,
                }
            )
            annotation_id += 1

    return {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": TEXT_CATEGORY_ID, "name": "text"}],
    }


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    values = sorted(values)
    index = (len(values) - 1) * fraction
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return values[lower]
    weight = index - lower
    return values[lower] * (1 - weight) + values[upper] * weight


def print_box_summary(records: list[dict[str, Any]], title: str) -> None:
    widths = []
    heights = []
    sqrt_areas = []
    aspects = []
    for record in records:
        for box in record["boxes"]:
            _, _, width, height = box["bbox"]
            widths.append(width)
            heights.append(height)
            sqrt_areas.append(math.sqrt(width * height))
            aspects.append(width / height if height > 0 else 0.0)

    print(f"\n{title}")
    print(f"  Images: {len(records)}")
    print(f"  Boxes:  {len(widths)}")
    if not widths:
        return
    print(
        "  sqrt(area) p10/p50/p90: "
        f"{percentile(sqrt_areas, 0.10):.1f} / {percentile(sqrt_areas, 0.50):.1f} / {percentile(sqrt_areas, 0.90):.1f}"
    )
    print(
        "  width p10/p50/p90: "
        f"{percentile(widths, 0.10):.1f} / {percentile(widths, 0.50):.1f} / {percentile(widths, 0.90):.1f}"
    )
    print(
        "  height p10/p50/p90: "
        f"{percentile(heights, 0.10):.1f} / {percentile(heights, 0.50):.1f} / {percentile(heights, 0.90):.1f}"
    )
    print(
        "  aspect ratio p10/p50/p90: "
        f"{percentile(aspects, 0.10):.2f} / {percentile(aspects, 0.50):.2f} / {percentile(aspects, 0.90):.2f}"
    )


def write_overlap_report(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "task_id",
                "file_name",
                "box_a_index",
                "box_b_index",
                "iou",
                "containment",
                "box_a",
                "box_b",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> int:
    args = parse_args()
    try:
        session, base_url = build_session(args.base_url)
        tasks = fetch_project_tasks(session, base_url, args.project_id)
    except Exception as exc:
        print(f"❌ {exc}")
        return 1

    print("=" * 70)
    print(f"Building text-only dataset from Label Studio project {args.project_id}")
    print("=" * 70)
    print(f"Fetched tasks: {len(tasks)}")

    records, overlap_rows, stats = build_task_records(
        tasks,
        args.dedup_iou_threshold,
        args.review_iou_threshold,
        args.review_containment_threshold,
    )

    if not records:
        print("❌ No annotated tasks with text boxes were found.")
        return 1

    images_dir = Path(args.images_dir)
    images_dir.mkdir(parents=True, exist_ok=True)
    try:
        missing_images = validate_and_download_images(
            session,
            base_url,
            records,
            images_dir,
            args.allow_missing_images,
            args.download_missing_images,
        )
    except Exception as exc:
        print(f"❌ {exc}")
        return 1

    train_records, val_records = split_records(records, args.val_ratio, args.seed)
    train_coco = build_coco(train_records)
    val_coco = build_coco(val_records)

    write_json(Path(args.output_train), train_coco)
    write_json(Path(args.output_val), val_coco)
    write_overlap_report(Path(args.overlap_report), overlap_rows)

    print(f"Annotated tasks kept: {stats['annotated_tasks']}")
    print(f"Skipped empty tasks:  {stats['skipped_empty_tasks']}")
    print(f"Deduped boxes removed: {stats['dedup_removed']}")
    print(f"Remaining suspicious overlaps reported: {len(overlap_rows)}")
    if missing_images:
        print(f"Missing local images: {len(missing_images)}")

    print_box_summary(train_records, "Train split")
    print_box_summary(val_records, "Validation split")

    print("\nOutputs")
    print(f"  Train JSON:   {args.output_train}")
    print(f"  Val JSON:     {args.output_val}")
    print(f"  Overlap CSV:  {args.overlap_report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())