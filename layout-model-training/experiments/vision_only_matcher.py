#!/usr/bin/env python3
"""
vision_only_matcher.py
──────────────────────
Pure VISION-ONLY baseline (no layout parser / no Detectron2).

A single SOTA vision model receives the raw yearbook page image and is
asked to:
  1. Locate every portrait photo on the page.
  2. Assign each a sequential number.
  3. Match each portrait to its adjacent text block.
  4. Return structured JSON identical to portrait_text_matcher.py output.
  5. Produce a bio summary.

Output: *_visiononly_matches.json, *_visiononly_table.png

Usage
-----
    source ~/.config/cborg.env && export OPENAI_API_KEY="$CBORG_API_KEY"
    python vision_only_matcher.py --image path/to/page.jpg
    python vision_only_matcher.py --model claude-opus-4-7   # override model
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT  = Path(__file__).resolve().parents[2]
OUTPUT_DIR = REPO_ROOT / "layout-model-training/experiments/portrait_matches"
DEFAULT_IMG = OUTPUT_DIR / "onlineRandomImage.png"
DEFAULT_MODEL = "claude-opus-4-7"   # most SOTA on CBORG as of May 2026

_FONT_REGULAR = "/usr/share/fonts/truetype/LiberationSans-Regular.ttf"
_FONT_BOLD    = "/usr/share/fonts/truetype/LiberationSans-Bold.ttf"

SYSTEM_PROMPT = """\
You are an expert yearbook archivist with exceptional visual acuity.

You will receive a raw yearbook page spread — NO bounding boxes or labels
have been added. You must do everything yourself:

STEP 1 — Find every portrait photograph on the page.
          Number them 1, 2, 3 … in natural reading order
          (left-to-right, top-to-bottom).

STEP 2 — For each portrait, record its bounding box as fractional coordinates
          of the full image dimensions (values between 0.0 and 1.0):
          x1=left edge, y1=top edge, x2=right edge, y2=bottom edge.

STEP 3 — Find the adjacent text block that describes that person
          (usually beside or below the photo).

STEP 4 — Return ONLY valid JSON — an array, one object per portrait:
[
  {
    "portrait_id": <int, starting at 1>,
    "bbox": {"x1": <float>, "y1": <float>, "x2": <float>, "y2": <float>},
    "name": "<full name as printed>",
    "nickname": "<nickname if present, else null>",
    "page_side": "<'left' or 'right'>",
    "activities": ["<activity>", ...],
    "quote": "<printed quote if present, else null>",
    "other": "<any remaining printed text not captured above, else null>",
    "bio": "<fluent 1-3 sentence summary using ONLY facts visibly printed — do not invent>"
  },
  ...
]

Rules:
- Include ALL portraits you can find.
- bbox must cover just the portrait photo, not the text block.
- Do NOT invent or infer — only transcribe what is visibly printed.
- Return raw JSON only — no markdown fences, no preamble.
"""


def to_b64(img_bgr: np.ndarray, max_side: int = 2000, quality: int = 90) -> str:
    h, w = img_bgr.shape[:2]
    if max(h, w) > max_side:
        scale = max_side / max(h, w)
        img_bgr = cv2.resize(img_bgr, (int(w * scale), int(h * scale)),
                             interpolation=cv2.INTER_AREA)
    _, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf.tobytes()).decode()


def _sanitize_json(raw: str) -> str:
    result  = []
    in_str  = False
    escaped = False
    for ch in raw:
        if escaped:
            result.append(ch)
            escaped = False
        elif ch == "\\" and in_str:
            result.append(ch)
            escaped = True
        elif ch == '"':
            in_str = not in_str
            result.append(ch)
        elif in_str and ord(ch) < 0x20:
            result.append(" ")
        else:
            result.append(ch)
    return "".join(result)


def call_vision_model(b64_image: str, model: str) -> list[dict]:
    try:
        from openai import OpenAI
    except ImportError:
        sys.exit("openai package not found.")

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        sys.exit("OPENAI_API_KEY is not set.")

    client = OpenAI(
        api_key=api_key,
        base_url=os.environ.get("OPENAI_BASE_URL"),
    )

    response = client.chat.completions.create(
        model=model,
        max_tokens=8192,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Here is the raw yearbook page. "
                            "Find all portrait photos, assign sequential numbers, "
                            "match each to its text block, and return JSON."
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{b64_image}",
                            "detail": "high",
                        },
                    },
                ],
            },
        ],
    )

    raw = response.choices[0].message.content.strip()
    print("\n── Vision-only raw response (first 2000 chars) ──")
    print(raw[:2000])
    print("─────────────────────────────────────────────────\n")

    if raw.startswith("```"):
        lines = raw.split("\n")
        raw   = "\n".join(lines[1:])
        if raw.rstrip().endswith("```"):
            raw = raw[: raw.rstrip().rfind("```")]
    raw = raw.strip()
    raw = _sanitize_json(raw)
    return json.loads(raw)


def _load_font(path: str, size: int):
    from PIL import ImageFont
    try:
        return ImageFont.truetype(path, size)
    except OSError:
        return ImageFont.load_default()


def _wrap_text(text: str, font, max_width: int, draw) -> list[str]:
    words   = text.split()
    lines   = []
    current = ""
    for word in words:
        test = (current + " " + word).strip()
        if draw.textlength(test, font=font) <= max_width:
            current = test
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines or [""]


def _crop_from_bbox(img_bgr: np.ndarray, bbox: dict, pad: int = 4, max_h: int = 140):
    """Crop portrait using fractional bbox returned by the vision model."""
    from PIL import Image as PilImage
    H, W = img_bgr.shape[:2]
    x1 = max(0, int(bbox["x1"] * W) - pad)
    y1 = max(0, int(bbox["y1"] * H) - pad)
    x2 = min(W, int(bbox["x2"] * W) + pad)
    y2 = min(H, int(bbox["y2"] * H) + pad)
    crop = img_bgr[y1:y2, x1:x2]
    ch, cw = crop.shape[:2]
    if ch > max_h:
        scale = max_h / ch
        crop  = cv2.resize(crop, (int(cw * scale), max_h), interpolation=cv2.INTER_AREA)
    return PilImage.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))


def save_png_table(output_dir: Path, stem: str,
                   records: list[dict],
                   img_bgr: np.ndarray) -> Path:
    from PIL import Image as PilImage, ImageDraw

    PAD        = 14
    NUM_COL_W  = 36
    PHOTO_W    = 120
    TEXT_COL_W = 540
    TOTAL_W    = PAD + NUM_COL_W + PAD + PHOTO_W + PAD + TEXT_COL_W + PAD
    FONT_SZ_NAME = 15
    FONT_SZ_BIO  = 13
    LINE_H_BIO   = FONT_SZ_BIO + 5

    fn_bold = _load_font(_FONT_BOLD,    FONT_SZ_NAME)
    fn_reg  = _load_font(_FONT_REGULAR, FONT_SZ_BIO)
    fn_num  = _load_font(_FONT_BOLD,    18)

    BG_EVEN = (252, 250, 246)
    BG_ODD  = (240, 237, 229)
    BG_HEAD = (51,  51,  51)
    C_NAME  = (20,  20,  60)
    C_BIO   = (40,  40,  40)
    C_MUTED = (100, 100, 100)
    C_WHITE = (255, 255, 255)
    DIVIDER = (200, 195, 185)

    scratch = PilImage.new("RGB", (TOTAL_W, 100))
    sdraw   = ImageDraw.Draw(scratch)

    row_data = []
    for rec in sorted(records, key=lambda r: r.get("portrait_id", 999)):
        pid   = rec.get("portrait_id", "?")
        name  = rec.get("name") or "UNKNOWN"
        nick  = rec.get("nickname")
        bio   = rec.get("bio") or ""
        bbox  = rec.get("bbox")
        if not bio:
            parts = list(rec.get("activities") or [])
            q = rec.get("quote")
            if q:
                parts.append(f'"{q}"')
            o = rec.get("other")
            if o:
                parts.append(o)
            bio = "  ".join(parts)

        # Crop portrait if bbox present
        crop = None
        if bbox and all(k in bbox for k in ("x1", "y1", "x2", "y2")):
            try:
                crop = _crop_from_bbox(img_bgr, bbox)
            except Exception:
                crop = None

        name_display = name + (f'  "{nick}"' if nick else "")
        lines_name = _wrap_text(name_display, fn_bold, TEXT_COL_W - PAD, sdraw)
        lines_bio  = _wrap_text(bio, fn_reg,  TEXT_COL_W - PAD, sdraw) if bio else []
        text_h = len(lines_name) * (FONT_SZ_NAME + 4) + len(lines_bio) * LINE_H_BIO
        crop_h = crop.height if crop else 0
        row_h  = max(text_h + PAD * 2, crop_h + PAD * 2, 50)
        row_data.append((pid, name_display, lines_name, lines_bio, crop, row_h))

    HEADER_H = 36
    total_h  = HEADER_H + sum(r[5] for r in row_data) + 1
    img_out  = PilImage.new("RGB", (TOTAL_W, total_h), BG_EVEN)
    draw     = ImageDraw.Draw(img_out)

    draw.rectangle([(0, 0), (TOTAL_W, HEADER_H)], fill=BG_HEAD)
    draw.text((PAD, 8), "#",           font=fn_num, fill=C_WHITE)
    draw.text((PAD + NUM_COL_W + PAD, 8), "Portrait", font=fn_num, fill=C_WHITE)
    draw.text((PAD + NUM_COL_W + PAD + PHOTO_W + PAD, 8),
              "Information (vision-only, no layout parser)", font=fn_num, fill=C_WHITE)

    y = HEADER_H
    for i, (pid, _nd, lines_name, lines_bio, crop, row_h) in enumerate(row_data):
        bg = BG_EVEN if i % 2 == 0 else BG_ODD
        draw.rectangle([(0, y), (TOTAL_W, y + row_h - 1)], fill=bg)
        draw.line([(0, y + row_h - 1), (TOTAL_W, y + row_h - 1)], fill=DIVIDER, width=1)

        # number
        num_x = PAD + (NUM_COL_W - draw.textlength(str(pid), font=fn_num)) // 2
        draw.text((num_x, y + (row_h - 22) // 2), str(pid), font=fn_num, fill=C_MUTED)

        # photo crop
        if crop:
            cx = PAD + NUM_COL_W + PAD
            # scale width to PHOTO_W if needed
            if crop.width > PHOTO_W:
                nh = int(crop.height * PHOTO_W / crop.width)
                crop = crop.resize((PHOTO_W, nh), PilImage.LANCZOS)
            cy = y + (row_h - crop.height) // 2
            img_out.paste(crop, (cx, cy))

        # text
        tx = PAD + NUM_COL_W + PAD + PHOTO_W + PAD
        ty = y + PAD
        for ln in lines_name:
            draw.text((tx, ty), ln, font=fn_bold, fill=C_NAME)
            ty += FONT_SZ_NAME + 4
        ty += 4
        for ln in lines_bio:
            draw.text((tx, ty), ln, font=fn_reg, fill=C_BIO)
            ty += LINE_H_BIO
        y += row_h

    out_path = output_dir / f"{stem}_visiononly_table.png"
    img_out.save(str(out_path), "PNG")
    return out_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image",  default=str(DEFAULT_IMG))
    p.add_argument("--model",  default=DEFAULT_MODEL)
    p.add_argument("--output-dir", default=str(OUTPUT_DIR))
    args = p.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    img_path = Path(args.image)
    if not img_path.exists():
        sys.exit(f"Image not found: {img_path}")

    img  = cv2.imread(str(img_path))
    h, w = img.shape[:2]
    stem = img_path.stem
    print(f"Image : {img_path.name}  ({w}×{h} px)")
    print(f"Model : {args.model}  [NO layout parser]")

    t0 = time.time()
    print("Encoding image and calling vision model …")
    b64     = to_b64(img)
    records = call_vision_model(b64, model=args.model)
    elapsed = time.time() - t0
    print(f"  → {len(records)} record(s) returned  ({elapsed:.1f}s)")

    json_path = output_dir / f"{stem}_visiononly_matches.json"
    json_path.write_text(json.dumps(records, indent=2, ensure_ascii=False))
    print(f"  JSON      : {json_path}")

    png_path = save_png_table(output_dir, stem, records, img)
    print(f"  PNG table : {png_path}")

    print(f"\n{'─'*65}")
    print(f"{'#':>3}  {'Name':<32}  {'Nickname':<14}  Side")
    print(f"{'─'*65}")
    for rec in sorted(records, key=lambda r: r.get("portrait_id", 999)):
        print(f"{rec.get('portrait_id','?'):>3}  "
              f"{rec.get('name','UNKNOWN'):<32}  "
              f"{(rec.get('nickname') or ''):<14}  "
              f"{rec.get('page_side','')}")
    print(f"{'─'*65}")
    print(f"\nTotal vision-model time (no layout parser): {elapsed:.1f}s")


if __name__ == "__main__":
    main()
