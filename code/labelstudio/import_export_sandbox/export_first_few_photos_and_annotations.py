#!/usr/bin/env python3
"""
Export the first few photo examples and their annotations from Label Studio.

This script:
1) Fetches tasks from a Label Studio project (paginated)
2) Keeps only photo labels: Student Photo, Faculty Photo, Group Photo
3) Downloads the corresponding source images
4) Saves extracted annotation metadata to JSON
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin, urlparse

import requests
from dotenv import load_dotenv

PHOTO_LABELS = {"Student Photo", "Faculty Photo", "Group Photo"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract first few photo images + annotations from Label Studio",
    )
    parser.add_argument(
        "--project-id",
        type=int,
        default=None,
        help="Label Studio project ID (overrides LABEL_STUDIO_PROJECT_ID)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of annotated photo tasks to extract",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="code/labelstudio/import_export_sandbox/output",
        help="Output directory for JSON + images",
    )
    parser.add_argument(
        "--page-size",
        type=int,
        default=100,
        help="Pagination page size when listing tasks",
    )
    return parser.parse_args()


def get_env_or_fail(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise ValueError(f"{name} is required")
    return value


def list_project_tasks(
    base_url: str,
    token: str,
    project_id: int,
    page_size: int,
) -> List[Dict[str, Any]]:
    all_tasks: List[Dict[str, Any]] = []
    page = 1

    while True:
        resp = requests.get(
            f"{base_url}/api/projects/{project_id}/tasks",
            params={"page": page, "page_size": page_size},
            headers={"Authorization": f"Token {token}"},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()

        if isinstance(data, list):
            tasks_page = data
            has_next = len(tasks_page) == page_size
        else:
            tasks_page = data.get("tasks", data.get("results", []))
            has_next = bool(data.get("next"))

        if not tasks_page:
            break

        all_tasks.extend(tasks_page)

        if not has_next:
            break

        page += 1

    return all_tasks


def build_image_url(base_url: str, image_ref: str) -> str:
    if image_ref.startswith("s3://"):
        # Convert s3://bucket/key -> https://bucket.s3.amazonaws.com/key
        without_scheme = image_ref[len("s3://") :]
        if "/" in without_scheme:
            bucket, key = without_scheme.split("/", 1)
            return f"https://{bucket}.s3.amazonaws.com/{key}"
        return image_ref
    if image_ref.startswith("http://") or image_ref.startswith("https://"):
        return image_ref
    return urljoin(f"{base_url}/", image_ref.lstrip("/"))


def choose_filename(task_id: int, image_url: str) -> str:
    parsed = urlparse(image_url)
    raw_name = Path(parsed.path).name or f"task_{task_id}.jpg"
    safe_name = raw_name.split("?")[0]
    return f"task_{task_id}_{safe_name}"


def extract_photo_boxes(task: Dict[str, Any]) -> List[Dict[str, Any]]:
    boxes: List[Dict[str, Any]] = []
    annotations = task.get("annotations", [])

    for annotation in annotations:
        for result in annotation.get("result", []):
            if result.get("type") != "rectanglelabels":
                continue

            value = result.get("value", {})
            labels = value.get("rectanglelabels", [])
            if not labels:
                continue

            label = labels[0]
            if label not in PHOTO_LABELS:
                continue

            boxes.append(
                {
                    "annotation_id": annotation.get("id"),
                    "created_at": annotation.get("created_at"),
                    "created_by": annotation.get("created_username"),
                    "label": label,
                    "x_pct": value.get("x"),
                    "y_pct": value.get("y"),
                    "width_pct": value.get("width"),
                    "height_pct": value.get("height"),
                    "original_width": result.get("original_width"),
                    "original_height": result.get("original_height"),
                }
            )

    return boxes


def download_image(url: str, token: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    headers = {}
    if "s3.amazonaws.com" not in url:
        headers = {"Authorization": f"Token {token}"}
    resp = requests.get(
        url,
        headers=headers,
        timeout=60,
    )
    resp.raise_for_status()
    output_path.write_bytes(resp.content)


def main() -> None:
    load_dotenv()
    args = parse_args()

    token = get_env_or_fail("LABEL_STUDIO_TOKEN")
    base_url = os.getenv("LABEL_STUDIO_BASE_URL", "https://app.humansignal.com").rstrip("/")

    project_id_val: Optional[int] = args.project_id
    if project_id_val is None:
        project_id_str = get_env_or_fail("LABEL_STUDIO_PROJECT_ID")
        project_id_val = int(project_id_str)

    out_dir = Path(args.out_dir)
    image_dir = out_dir / "images"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Fetching tasks from project {project_id_val}...")
    tasks = list_project_tasks(
        base_url=base_url,
        token=token,
        project_id=project_id_val,
        page_size=args.page_size,
    )
    print(f"Found {len(tasks)} tasks")

    extracted: List[Dict[str, Any]] = []

    for task in tasks:
        if len(extracted) >= args.limit:
            break

        photo_boxes = extract_photo_boxes(task)
        if not photo_boxes:
            continue

        task_id = task.get("id")
        image_ref = task.get("data", {}).get("image", "")
        image_url = build_image_url(base_url, image_ref)

        filename = choose_filename(task_id=task_id, image_url=image_url)
        local_image_path = image_dir / filename

        try:
            download_image(image_url, token=token, output_path=local_image_path)
            image_saved = True
            image_error = None
        except Exception as ex:  # noqa: BLE001
            image_saved = False
            image_error = str(ex)

        extracted.append(
            {
                "task_id": task_id,
                "source_image_ref": image_ref,
                "source_image_url": image_url,
                "local_image_path": str(local_image_path),
                "image_saved": image_saved,
                "image_error": image_error,
                "photo_annotations": photo_boxes,
            }
        )

    output_json = out_dir / "first_few_photo_annotations.json"
    output_json.write_text(json.dumps(extracted, indent=2), encoding="utf-8")

    print("\nDone.")
    print(f"Extracted tasks: {len(extracted)}")
    print(f"Saved annotations: {output_json}")
    print(f"Saved images dir: {image_dir}")


if __name__ == "__main__":
    main()
