#!/usr/bin/env python3
"""Create a fresh text-only COCO dataset for specific Label Studio task IDs.

This script re-fetches the latest annotations from Label Studio, filters to the
requested yearbook page task IDs, and keeps only the text labels used for text
training: `Name` and `Additional Text`.

Output is written to a separate COCO JSON file so existing datasets remain
untouched.
"""

import argparse
import base64
import json
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple
from urllib.parse import parse_qs, urlparse

try:
    import requests
except ImportError:
    requests = None

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

TEXT_LABELS = {"Name", "Additional Text"}
TEXT_CATEGORY_ID = 1
DEDUP_IOU_THRESHOLD = 0.85
DEFAULT_PROJECT_ID = 158111
DEFAULT_OUTPUT = "layout-model-training/data/yearbook/annotations-text-only-task-ids.json"
DEFAULT_IMAGES_DIR = "layout-model-training/data/yearbook/images"
DEFAULT_BASE_URL = "https://app.humansignal.com"
DEFAULT_FALLBACK_DIMENSION = 2048


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch fresh annotations for specific Label Studio task IDs and "
            "build a text-only COCO dataset using only Name and Additional Text."
        )
    )
    parser.add_argument(
        "task_ids",
        nargs="*",
        help="Label Studio task IDs. You can pass them space-separated and/or comma-separated.",
    )
    parser.add_argument(
        "--task-id-file",
        help="Optional file containing task IDs separated by commas, spaces, or newlines.",
    )
    parser.add_argument(
        "--project-id",
        type=int,
        default=DEFAULT_PROJECT_ID,
        help="Expected Label Studio project ID for validation.",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help="Path to write the new COCO annotation file.",
    )
    parser.add_argument(
        "--images-dir",
        default=DEFAULT_IMAGES_DIR,
        help="Local image directory to validate file_name presence against.",
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv("LABEL_STUDIO_BASE_URL", DEFAULT_BASE_URL),
        help="Label Studio base URL.",
    )
    parser.add_argument(
        "--allow-missing-images",
        action="store_true",
        help="Do not fail if some referenced images are not present locally.",
    )
    parser.add_argument(
        "--download-images",
        action="store_true",
        help="Download missing images from Label Studio before writing the annotation file.",
    )
    return parser.parse_args()


def parse_task_id_tokens(tokens: Iterable[str]) -> List[int]:
    task_ids = []
    seen = set()
    for token in tokens:
        for piece in token.replace(",", " ").split():
            value = int(piece)
            if value not in seen:
                seen.add(value)
                task_ids.append(value)
    return task_ids


def load_task_ids(args: argparse.Namespace) -> List[int]:
    tokens = list(args.task_ids)
    if args.task_id_file:
        file_text = Path(args.task_id_file).read_text(encoding="utf-8")
        tokens.append(file_text)

    if not tokens:
        raise ValueError("Provide at least one task ID or use --task-id-file.")

    task_ids = parse_task_id_tokens(tokens)
    if not task_ids:
        raise ValueError("No valid task IDs were found in the provided input.")
    return task_ids


def build_session(base_url: str) -> Tuple[object, str]:
    if requests is None:
        raise RuntimeError("The 'requests' package is required. Install dependencies from requirements.txt.")

    if load_dotenv is not None:
        load_dotenv()

    api_token = os.getenv("LABEL_STUDIO_TOKEN")
    if not api_token:
        raise RuntimeError("LABEL_STUDIO_TOKEN not found in environment or .env")

    session = requests.Session()
    session.headers.update({"Authorization": "Token {0}".format(api_token)})
    return session, base_url.rstrip("/")


def fetch_task(session: object, base_url: str, task_id: int) -> Dict:
    response = session.get("{0}/api/tasks/{1}".format(base_url, task_id), timeout=60)
    if response.status_code != 200:
        raise RuntimeError(
            "Failed to fetch task {0}: HTTP {1} {2}".format(
                task_id, response.status_code, response.text[:300]
            )
        )
    return response.json()


def get_task_dimensions(task: Dict, results: Sequence[Dict]) -> Tuple[int, int]:
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


def bbox_iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
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

    union = (aw * ah) + (bw * bh) - inter
    if union <= 0:
        return 0.0
    return inter / union


def deduplicate_boxes(boxes: Sequence[Dict]) -> List[Dict]:
    deduped = []
    for box in sorted(boxes, key=lambda entry: -entry["area"]):
        duplicate = False
        for existing in deduped:
            if bbox_iou(existing["bbox"], box["bbox"]) >= DEDUP_IOU_THRESHOLD:
                duplicate = True
                break
        if not duplicate:
            deduped.append(box)
    return deduped


def decode_fileuri(fileuri_b64: str) -> str:
    """Decode a base64-encoded S3/file URI and return just the filename."""
    # Fix missing base64 padding
    padding = 4 - len(fileuri_b64) % 4
    if padding != 4:
        fileuri_b64 += "=" * padding
    decoded = base64.b64decode(fileuri_b64).decode("utf-8")
    # decoded is e.g. s3://bucket/path/to/file.jpg
    return decoded.rsplit("/", 1)[-1]


def extract_image_url(task: Dict) -> str:
    """Return the raw image URL stored in the task data field."""
    return task.get("data", {}).get("image", "")


def extract_filename(task: Dict) -> str:
    image_url = extract_image_url(task)
    if not image_url:
        return "task_{0}.jpg".format(task["id"])

    # Label Studio individual-task API returns proxy URLs like:
    #   /data/upload/...  or  ?fileuri=<base64_encoded_s3_path>
    parsed = urlparse(image_url)
    qs = parse_qs(parsed.query)

    if "fileuri" in qs:
        try:
            return decode_fileuri(qs["fileuri"][0])
        except Exception:
            pass

    # Normal URL — take the last path component
    path = parsed.path
    if path and "/" in path:
        name = path.rsplit("/", 1)[-1]
        if name:
            return name

    if "/" in image_url:
        return image_url.rsplit("/", 1)[-1]

    return "task_{0}.jpg".format(task["id"])


def collect_text_boxes(task: Dict) -> Tuple[List[Dict], int, int, Dict[str, int]]:
    annotations = task.get("annotations", [])
    all_results = []
    for annotation in annotations:
        all_results.extend(annotation.get("result", []))

    if not all_results:
        return [], DEFAULT_FALLBACK_DIMENSION, DEFAULT_FALLBACK_DIMENSION, {}

    width, height = get_task_dimensions(task, all_results)
    boxes = []
    counts = {label: 0 for label in TEXT_LABELS}

    for result in all_results:
        if result.get("type") != "rectanglelabels":
            continue
        value = result.get("value", {})
        labels = value.get("rectanglelabels", [])
        if not labels:
            continue
        label = labels[0]
        if label not in TEXT_LABELS:
            continue

        img_w = result.get("original_width", width) or width
        img_h = result.get("original_height", height) or height

        x_pct = value.get("x", 0)
        y_pct = value.get("y", 0)
        w_pct = value.get("width", 0)
        h_pct = value.get("height", 0)

        x_abs = (x_pct / 100.0) * img_w
        y_abs = (y_pct / 100.0) * img_h
        w_abs = (w_pct / 100.0) * img_w
        h_abs = (h_pct / 100.0) * img_h

        if w_abs <= 0 or h_abs <= 0:
            continue

        boxes.append(
            {
                "bbox": [x_abs, y_abs, w_abs, h_abs],
                "area": w_abs * h_abs,
                "source_label": label,
            }
        )
        counts[label] = counts.get(label, 0) + 1

    return deduplicate_boxes(boxes), width, height, counts


def validate_project(task: Dict, expected_project_id: int) -> None:
    project_id = task.get("project")
    if project_id is not None and int(project_id) != int(expected_project_id):
        raise RuntimeError(
            "Task {0} belongs to project {1}, expected {2}".format(
                task.get("id"), project_id, expected_project_id
            )
        )


def download_image(
    session: object,
    base_url: str,
    image_url: str,
    output_path: Path,
) -> None:
    """Download one image through the Label Studio proxy."""
    # If the URL is relative / a proxy path, prefix the base URL
    if image_url.startswith("http://") or image_url.startswith("https://"):
        full_url = image_url
    else:
        full_url = "{0}/{1}".format(base_url.rstrip("/"), image_url.lstrip("/"))

    response = session.get(full_url, timeout=120)  # type: ignore[union-attr]
    response.raise_for_status()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(response.content)


def validate_and_download_images(
    session: object,
    base_url: str,
    tasks: Sequence[Dict],
    images_dir: Path,
    allow_missing: bool,
    download: bool,
) -> List[str]:
    """Check local image presence; optionally download missing ones."""
    task_by_filename = {extract_filename(t): t for t in tasks}
    missing_after = []

    for filename, task in task_by_filename.items():
        dest = images_dir / filename
        if dest.exists():
            continue
        if download:
            image_url = extract_image_url(task)
            if not image_url:
                missing_after.append(filename)
                continue
            try:
                print("  Downloading {0}...".format(filename))
                download_image(session, base_url, image_url, dest)
            except Exception as exc:
                print("  ⚠ Failed to download {0}: {1}".format(filename, exc))
                missing_after.append(filename)
        else:
            missing_after.append(filename)

    if missing_after and not allow_missing:
        preview = ", ".join(missing_after[:10])
        raise RuntimeError(
            "Missing {0} image files in {1}: {2}{3}".format(
                len(missing_after),
                images_dir,
                preview,
                " ..." if len(missing_after) > 10 else "",
            )
        )
    return missing_after


def build_coco_dataset(tasks: Sequence[Dict]) -> Tuple[Dict, Dict[str, int]]:
    images = []
    annotations = []
    annotation_id = 1
    label_totals = {label: 0 for label in TEXT_LABELS}

    for image_id, task in enumerate(tasks, start=1):
        text_boxes, width, height, counts = collect_text_boxes(task)
        if not text_boxes:
            continue

        for label, count in counts.items():
            label_totals[label] = label_totals.get(label, 0) + count

        images.append(
            {
                "id": image_id,
                "file_name": extract_filename(task),
                "width": width,
                "height": height,
                "task_id": task["id"],
            }
        )

        for box in text_boxes:
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

    coco = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": TEXT_CATEGORY_ID, "name": "text"}],
    }
    return coco, label_totals


def main() -> int:
    args = parse_args()

    try:
        task_ids = load_task_ids(args)
        session, base_url = build_session(args.base_url)
    except Exception as exc:
        print("❌ {0}".format(exc))
        return 1

    print("=" * 70)
    print("Fetching fresh text annotations for selected task IDs")
    print("=" * 70)
    print("Task IDs: {0}".format(", ".join(str(task_id) for task_id in task_ids)))

    fetched_tasks = []
    for task_id in task_ids:
        print("Fetching task {0}...".format(task_id))
        try:
            task = fetch_task(session, base_url, task_id)
            validate_project(task, args.project_id)
        except Exception as exc:
            print("❌ {0}".format(exc))
            return 1
        fetched_tasks.append(task)

    coco_data, label_totals = build_coco_dataset(fetched_tasks)
    if not coco_data["images"]:
        print("❌ None of the selected tasks contained Name or Additional Text boxes.")
        return 1

    images_dir = Path(args.images_dir)
    images_dir.mkdir(parents=True, exist_ok=True)
    try:
        if args.download_images:
            print("\nDownloading missing images...")
        missing_images = validate_and_download_images(
            session,
            base_url,
            fetched_tasks,
            images_dir,
            args.allow_missing_images,
            args.download_images,
        )
    except Exception as exc:
        print("❌ {0}".format(exc))
        return 1

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(coco_data, indent=2), encoding="utf-8")

    print("\n✅ Wrote new dedicated annotation file")
    print("   Output: {0}".format(output_path))
    print("   Images: {0}".format(len(coco_data["images"])))
    print("   Text boxes: {0}".format(len(coco_data["annotations"])))
    print("   Name boxes: {0}".format(label_totals.get("Name", 0)))
    print("   Additional Text boxes: {0}".format(label_totals.get("Additional Text", 0)))
    if missing_images:
        print("   Missing local images: {0}".format(len(missing_images)))
    print("\nOlder annotation files were not modified.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
