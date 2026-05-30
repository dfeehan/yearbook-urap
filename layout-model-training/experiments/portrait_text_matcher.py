#!/usr/bin/env python3
"""
portrait_text_matcher.py
────────────────────────
Pipeline
  1. LAYOUT PARSER (custom trained Detectron2 Faster-RCNN R50-FPN, human-figure class)
     → detects all portrait boxes on the yearbook page.
  2. Draw numbered coloured boxes on a copy of the page image.
  3. VISION MODEL (GPT-4o via CBORG gateway)
     → receives the numbered page; matches each portrait number to the
       adjacent text block; returns structured JSON.
  4. Save  *_numbered.jpg,  *_annotated.jpg,  *_matches.json.

Usage
-----
    source ~/.config/cborg.env && export OPENAI_API_KEY="$CBORG_API_KEY"
    python portrait_text_matcher.py                         # default image
    python portrait_text_matcher.py --image path/to/page.jpg
    python portrait_text_matcher.py --threshold 0.6         # stricter detection
    python portrait_text_matcher.py --model gpt-4o-mini     # cheaper model
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT  = Path(__file__).resolve().parents[2]
OUTPUT_DIR = REPO_ROOT / "layout-model-training/experiments/portrait_matches"
DATA_DIR   = REPO_ROOT / "layout-model-training/data/yearbook/images"
DEFAULT_IMG = DATA_DIR / "snipscuts19401940cent_pages_28_29.jpg"

DEFAULT_MODEL_WEIGHTS = str(
    REPO_ROOT / "layout-model-training/outputs/yearbook-small/fast_rcnn_R_50_FPN_3x/model_final.pth"
)
DEFAULT_MODEL_CONFIG = str(
    REPO_ROOT / "layout-model-training/outputs/yearbook-small/fast_rcnn_R_50_FPN_3x/config.yaml"
)

# ── vision model prompt ────────────────────────────────────────────────────────
SYSTEM_PROMPT = """\
You are an expert yearbook archivist.
The user will show you a yearbook page spread. Every detected portrait photo
has been outlined with a coloured rectangle and labelled with a bold number
(1, 2, 3 …) in the top-left corner of the box.

Your task:
  For EVERY numbered portrait, find the text block on the SAME page that
  corresponds to that person (usually the text block closest to, beside, or
  below the portrait — containing the person's name and biography).

Return ONLY valid JSON — an array, one object per portrait:
[
  {
    "portrait_id": <int matching the number on the box>,
    "name": "<full name as printed>",
    "nickname": "<nickname in quotes if present, else null>",
    "page_side": "<'left' or 'right'>",
    "activities": ["<activity 1>", ...],
    "quote": "<printed quote if present, else null>",
    "other": "<any remaining printed text not captured above, else null>",
    "bio": "<a clear, fluent 1-3 sentence summary of all information about this person — reorganise for readability but use ONLY facts visibly printed on the page, do not invent anything>"
  },
  ...
]

Rules:
- portrait_id must exactly match the number shown on the coloured box.
- Include ALL numbered portraits, even if you cannot find matching text
  (set name to "UNKNOWN" and bio to null in that case).
- Do NOT invent or infer — only use what is visibly printed on the page.
- Return raw JSON only — no markdown fences, no preamble.
"""

# ── Step 1: Portrait detection via custom-trained human detector ──────────────

def build_portrait_predictor(threshold: float = 0.5,
                             weights: str = DEFAULT_MODEL_WEIGHTS,
                             config_file: str = DEFAULT_MODEL_CONFIG):
    """Return a Detectron2 DefaultPredictor using the custom human-figure model."""
    from detectron2.config import get_cfg
    from detectron2.engine import DefaultPredictor

    if not Path(weights).exists():
        sys.exit(f"Model weights not found: {weights}")
    if not Path(config_file).exists():
        sys.exit(f"Model config not found: {config_file}")

    cfg = get_cfg()
    cfg.merge_from_file(config_file)
    cfg.MODEL.WEIGHTS = weights
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold
    cfg.MODEL.DEVICE = "cpu"
    cfg.freeze()
    return DefaultPredictor(cfg)


def detect_portraits(img_bgr: np.ndarray, predictor) -> list[list[int]]:
    """
    Run the human-figure detector and return boxes sorted in reading order.
    Returns list of [x1, y1, x2, y2].
    """
    outputs   = predictor(img_bgr)
    instances = outputs["instances"].to("cpu")

    # Custom model has a single class (human_figure = 0); take all predictions
    boxes   = instances.pred_boxes.tensor.numpy()
    scores  = instances.scores.numpy()

    if len(boxes) == 0:
        return []

    # Containment NMS: suppress boxes ≥70% inside a larger box
    boxes = _containment_nms(boxes, thresh=0.70)

    h, _ = img_bgr.shape[:2]
    # Sort reading order: row-major (bucket y into rows), then left-to-right
    boxes.sort(key=lambda b: (b[1] // (h // 8), b[0]))
    return [[int(v) for v in b] for b in boxes]


def _containment_nms(boxes: np.ndarray, thresh: float = 0.70) -> list:
    if len(boxes) == 0:
        return []
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    order = np.argsort(-areas)
    keep, suppressed = [], set()
    for i_pos, i in enumerate(order):
        if i in suppressed:
            continue
        keep.append(i)
        for j in order[i_pos + 1:]:
            if j in suppressed:
                continue
            ix1 = max(boxes[i, 0], boxes[j, 0])
            iy1 = max(boxes[i, 1], boxes[j, 1])
            ix2 = min(boxes[i, 2], boxes[j, 2])
            iy2 = min(boxes[i, 3], boxes[j, 3])
            inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
            if inter / (areas[j] + 1e-6) >= thresh:
                suppressed.add(j)
    return boxes[keep].tolist()


# ── Step 2: Draw numbered portrait boxes ─────────────────────────────────────

COLOURS = [
    (220,  50,  50), ( 50, 160,  50), ( 50,  50, 220),
    (200, 130,   0), (150,   0, 200), (  0, 180, 180),
    (200,   0, 120), (100, 100,   0), (  0, 120, 200),
    (180,  90,   0),
]


def draw_numbered_portraits(img_bgr: np.ndarray, boxes: list[list[int]]) -> np.ndarray:
    """Overlay numbered coloured boxes on the image."""
    out = img_bgr.copy()
    h, _ = out.shape[:2]
    font  = cv2.FONT_HERSHEY_DUPLEX
    thick = max(3, h // 400)

    for rank, box in enumerate(boxes, 1):
        x1, y1, x2, y2 = box
        colour = COLOURS[(rank - 1) % len(COLOURS)]

        # Box outline
        cv2.rectangle(out, (x1, y1), (x2, y2), colour, thick)

        # Number badge (white fill, coloured border)
        label      = str(rank)
        badge_h    = max(36, h // 50)
        font_scale = badge_h / 42.0
        font_thick = max(2, thick - 1)
        (tw, th), _ = cv2.getTextSize(label, font, font_scale, font_thick)
        pad = 6
        bx2 = x1 + tw + pad * 2
        by2 = y1 + th + pad * 2
        cv2.rectangle(out, (x1, y1), (bx2, by2), (255, 255, 255), -1)
        cv2.rectangle(out, (x1, y1), (bx2, by2), colour, thick)
        cv2.putText(out, label, (x1 + pad, y1 + th + pad),
                    font, font_scale, colour, font_thick, cv2.LINE_AA)

    return out


# ── Step 3: Vision model matching ─────────────────────────────────────────────

def to_b64(img_bgr: np.ndarray, max_side: int = 2000, quality: int = 90) -> str:
    h, w = img_bgr.shape[:2]
    if max(h, w) > max_side:
        scale = max_side / max(h, w)
        img_bgr = cv2.resize(img_bgr, (int(w * scale), int(h * scale)),
                             interpolation=cv2.INTER_AREA)
    _, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf.tobytes()).decode()


def call_vision_model(b64_image: str, model: str = "gpt-4o") -> list[dict]:
    try:
        from openai import OpenAI
    except ImportError:
        sys.exit("openai package not found.  Run: pip install openai")

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        sys.exit("OPENAI_API_KEY is not set.")

    client = OpenAI(
        api_key=api_key,
        base_url=os.environ.get("OPENAI_BASE_URL"),
    )

    response = client.chat.completions.create(
        model=model,
        max_tokens=4096,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Here is the yearbook page with numbered portraits. "
                            "Match each portrait number to its text block and return JSON."
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
    print("\n── Vision model raw response ──")
    print(raw[:3000])
    print("──────────────────────────────\n")

    # Strip markdown fences (```json ... ``` or ``` ... ```)
    if raw.startswith("```"):
        lines = raw.split("\n")
        raw   = "\n".join(lines[1:])
        if raw.rstrip().endswith("```"):
            raw = raw[: raw.rstrip().rfind("```")]
    raw = raw.strip()

    # Sanitize literal control characters inside JSON string values.
    # GPT-4o sometimes embeds raw newlines/tabs in strings (invalid JSON).
    raw = _sanitize_json(raw)

    return json.loads(raw)


def _sanitize_json(raw: str) -> str:
    """Replace unescaped control characters inside JSON string literals with a space."""
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
            result.append(" ")   # replace raw control char with space
        else:
            result.append(ch)
    return "".join(result)


# ── Step 4: Annotate final image with names ───────────────────────────────────

def draw_names(img_bgr: np.ndarray,
               boxes: list[list[int]],
               records: list[dict]) -> np.ndarray:
    out  = img_bgr.copy()
    h, w = out.shape[:2]
    font = cv2.FONT_HERSHEY_DUPLEX
    name_map = {r["portrait_id"]: r.get("name", "") for r in records
                if "portrait_id" in r}

    for rank, box in enumerate(boxes, 1):
        name = name_map.get(rank, "")
        if not name or name == "UNKNOWN":
            continue
        x1, y1, x2, y2 = box
        bw    = x2 - x1
        fs    = max(0.45, min(1.1, bw / 280.0))
        ft    = max(1, int(fs * 2))
        (tw, th), _ = cv2.getTextSize(name, font, fs, ft)
        lx1, ly1 = x1, y2 + 2
        lx2, ly2 = min(w, x1 + tw + 8), y2 + th + 10
        cv2.rectangle(out, (lx1, ly1), (lx2, ly2), (255, 255, 255), -1)
        cv2.putText(out, name, (lx1 + 4, ly2 - 4),
                    font, fs, (20, 20, 20), ft, cv2.LINE_AA)

    return out


# ── Step 5: PIL composite PNG results table ──────────────────────────────────

_FONT_REGULAR = "/usr/share/fonts/truetype/LiberationSans-Regular.ttf"
_FONT_BOLD    = "/usr/share/fonts/truetype/LiberationSans-Bold.ttf"
_FONT_ITALIC  = "/usr/share/fonts/truetype/LiberationSans-Italic.ttf"


def _load_font(path: str, size: int):
    from PIL import ImageFont
    try:
        return ImageFont.truetype(path, size)
    except OSError:
        return ImageFont.load_default()


def _crop_portrait(img_bgr: np.ndarray, box: list[int],
                   pad: int = 4, max_h: int = 160):
    """Crop a portrait box from the image and return as PIL RGB image."""
    from PIL import Image as PilImage
    H, W = img_bgr.shape[:2]
    x1, y1, x2, y2 = box
    x1 = max(0, x1 - pad); y1 = max(0, y1 - pad)
    x2 = min(W, x2 + pad); y2 = min(H, y2 + pad)
    crop = img_bgr[y1:y2, x1:x2]
    ch, cw = crop.shape[:2]
    if ch > max_h:
        scale = max_h / ch
        crop  = cv2.resize(crop, (int(cw * scale), max_h),
                           interpolation=cv2.INTER_AREA)
    return PilImage.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))


def _wrap_text(text: str, font, max_width: int, draw) -> list[str]:
    """Word-wrap text to fit within max_width pixels."""
    words  = text.split()
    lines  = []
    current = ""
    for word in words:
        test = (current + " " + word).strip()
        w = draw.textlength(test, font=font)
        if w <= max_width:
            current = test
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines or [""]


def save_png_table(output_dir: Path, stem: str,
                   boxes: list[list[int]], records: list[dict],
                   img_bgr: np.ndarray) -> Path:
    """Render a composite PNG: portrait crop | number | bio/info per row."""
    from PIL import Image as PilImage, ImageDraw, ImageFont

    id_to_box = {i + 1: b for i, b in enumerate(boxes)}
    id_to_rec = {r["portrait_id"]: r for r in records if "portrait_id" in r}

    # ── layout constants ──────────────────────────────────────────────────────
    PAD        = 14
    PHOTO_W    = 120        # target width for portrait crop column
    TEXT_COL_W = 560        # width for the info text column
    NUM_COL_W  = 36
    TOTAL_W    = PAD + NUM_COL_W + PAD + PHOTO_W + PAD + TEXT_COL_W + PAD
    FONT_SZ_NAME = 15
    FONT_SZ_BIO  = 13
    FONT_SZ_SMALL= 12
    LINE_H_BIO   = FONT_SZ_BIO + 5

    fn_bold   = _load_font(_FONT_BOLD,    FONT_SZ_NAME)
    fn_reg    = _load_font(_FONT_REGULAR, FONT_SZ_BIO)
    fn_italic = _load_font(_FONT_ITALIC,  FONT_SZ_SMALL)
    fn_num    = _load_font(_FONT_BOLD,    18)

    BG_EVEN  = (252, 250, 246)
    BG_ODD   = (240, 237, 229)
    BG_HEAD  = (51,  51,  51)
    C_NAME   = (20,  20,  60)
    C_BIO    = (40,  40,  40)
    C_MUTED  = (100, 100, 100)
    C_WHITE  = (255, 255, 255)
    DIVIDER  = (200, 195, 185)

    # ── first pass: measure row heights ──────────────────────────────────────
    # Use a scratch image for text measurement
    scratch = PilImage.new("RGB", (TOTAL_W, 100))
    sdraw   = ImageDraw.Draw(scratch)

    row_data = []   # list of (pid, crop_pil, lines_name, lines_bio, row_h)
    for pid in sorted(id_to_box.keys()):
        box  = id_to_box[pid]
        rec  = id_to_rec.get(pid, {})
        crop = _crop_portrait(img_bgr, box, max_h=140)

        name = rec.get("name") or "UNKNOWN"
        bio  = rec.get("bio")  or ""

        # Supplement bio with any fields that might be missing from it
        # (activities, quote, other) — appended as plain fallback if bio absent
        if not bio:
            parts = []
            acts = rec.get("activities") or []
            if acts:
                parts.append("Activities: " + "; ".join(acts))
            q = rec.get("quote")
            if q:
                parts.append(f'"{q}"')
            o = rec.get("other")
            if o:
                parts.append(o)
            bio = "  ".join(parts)

        nick = rec.get("nickname")
        name_display = name + (f'  "{nick}"' if nick else "")

        lines_name = _wrap_text(name_display, fn_bold, TEXT_COL_W - PAD, sdraw)
        lines_bio  = _wrap_text(bio, fn_reg,  TEXT_COL_W - PAD, sdraw) if bio else []

        text_h  = len(lines_name) * (FONT_SZ_NAME + 4) + len(lines_bio) * LINE_H_BIO
        row_h   = max(crop.height + PAD * 2, text_h + PAD * 2, 60)
        row_data.append((pid, crop, name_display, lines_name, lines_bio, row_h))

    # ── second pass: render ───────────────────────────────────────────────────
    HEADER_H = 36
    total_h  = HEADER_H + sum(r[5] for r in row_data) + 1
    img_out  = PilImage.new("RGB", (TOTAL_W, total_h), BG_EVEN)
    draw     = ImageDraw.Draw(img_out)

    # header bar
    draw.rectangle([(0, 0), (TOTAL_W, HEADER_H)], fill=BG_HEAD)
    draw.text((PAD, 8), "#",            font=fn_num,  fill=C_WHITE)
    draw.text((PAD + NUM_COL_W + PAD, 8), "Portrait", font=fn_num, fill=C_WHITE)
    draw.text((PAD + NUM_COL_W + PAD + PHOTO_W + PAD, 8), "Information",
              font=fn_num, fill=C_WHITE)

    y = HEADER_H
    for i, (pid, crop, _name_display, lines_name, lines_bio, row_h) in enumerate(row_data):
        bg = BG_EVEN if i % 2 == 0 else BG_ODD
        draw.rectangle([(0, y), (TOTAL_W, y + row_h - 1)], fill=bg)
        draw.line([(0, y + row_h - 1), (TOTAL_W, y + row_h - 1)], fill=DIVIDER, width=1)

        # number
        num_x = PAD + (NUM_COL_W - draw.textlength(str(pid), font=fn_num)) // 2
        num_y = y + (row_h - 22) // 2
        draw.text((num_x, num_y), str(pid), font=fn_num, fill=C_MUTED)

        # portrait crop (no border)
        cx = PAD + NUM_COL_W + PAD
        cy = y + (row_h - crop.height) // 2
        # Scale crop width to PHOTO_W
        scale_w = PHOTO_W / crop.width if crop.width > PHOTO_W else 1.0
        if scale_w < 1.0:
            new_w = PHOTO_W
            new_h = int(crop.height * scale_w)
            crop  = crop.resize((new_w, new_h), PilImage.LANCZOS)
            cy    = y + (row_h - new_h) // 2
        img_out.paste(crop, (cx, cy))

        # text
        tx  = PAD + NUM_COL_W + PAD + PHOTO_W + PAD
        ty  = y + PAD
        for ln in lines_name:
            draw.text((tx, ty), ln, font=fn_bold, fill=C_NAME)
            ty += FONT_SZ_NAME + 4
        ty += 4   # small gap between name and bio
        for ln in lines_bio:
            draw.text((tx, ty), ln, font=fn_reg, fill=C_BIO)
            ty += LINE_H_BIO

        y += row_h

    out_path = output_dir / f"{stem}_table.png"
    img_out.save(str(out_path), "PNG")
    return out_path


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--image",         default=str(DEFAULT_IMG))
    p.add_argument("--threshold",     type=float, default=0.5,
                   help="Detectron2 score threshold for human detection (default 0.5)")
    p.add_argument("--model",         default="gpt-4o",
                   help="Vision model name (default gpt-4o)")
    p.add_argument("--model-weights", default=DEFAULT_MODEL_WEIGHTS,
                   help="Path to custom human detector weights (.pth)")
    p.add_argument("--model-config",  default=DEFAULT_MODEL_CONFIG,
                   help="Path to custom human detector config (.yaml)")
    p.add_argument("--output-dir",    default=str(OUTPUT_DIR))
    return p.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    img_path = Path(args.image)
    if not img_path.exists():
        sys.exit(f"Image not found: {img_path}")

    img  = cv2.imread(str(img_path))
    h, w = img.shape[:2]
    stem = img_path.stem
    print(f"Image : {img_path.name}  ({w}×{h} px)")

    # ── 1. Portrait detection ─────────────────────────────────────────────────
    print(f"Loading custom human detector (threshold={args.threshold}) …")
    print(f"  weights: {args.model_weights}")
    print(f"  config : {args.model_config}")
    predictor = build_portrait_predictor(
        threshold=args.threshold,
        weights=args.model_weights,
        config_file=args.model_config,
    )

    print("Detecting portrait regions …")
    boxes = detect_portraits(img, predictor)
    print(f"  → {len(boxes)} portrait(s) found")

    if len(boxes) == 0:
        print("No portraits detected. Try lowering --threshold (e.g. --threshold 0.3).")
        sys.exit(1)

    # ── 2. Draw numbered boxes ────────────────────────────────────────────────
    numbered_img  = draw_numbered_portraits(img, boxes)
    numbered_path = output_dir / f"{stem}_numbered.jpg"
    cv2.imwrite(str(numbered_path), numbered_img)
    print(f"  Numbered image : {numbered_path}")

    # ── 3. Vision model matching ──────────────────────────────────────────────
    print(f"Calling vision model ({args.model}) to match portraits ↔ text …")
    b64     = to_b64(numbered_img)
    records = call_vision_model(b64, model=args.model)
    print(f"  → {len(records)} record(s) returned")

    # Attach bounding box coordinates to every record
    for rec in records:
        pid = rec.get("portrait_id")
        if isinstance(pid, int) and 1 <= pid <= len(boxes):
            b = boxes[pid - 1]
            rec["bbox"] = {"x1": b[0], "y1": b[1], "x2": b[2], "y2": b[3]}

    # ── 4. Save outputs ───────────────────────────────────────────────────────
    json_path = output_dir / f"{stem}_matches.json"
    json_path.write_text(json.dumps(records, indent=2, ensure_ascii=False))
    print(f"  JSON           : {json_path}")

    annotated      = draw_names(numbered_img, boxes, records)
    annotated_path = output_dir / f"{stem}_annotated.jpg"
    cv2.imwrite(str(annotated_path), annotated)
    print(f"  Annotated image: {annotated_path}")

    # ── PNG results table ─────────────────────────────────────────────────────
    png_path = save_png_table(output_dir, stem, boxes, records, img)
    print(f"  PNG table      : {png_path}")

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"\n{'─'*70}")
    print(f"{'#':>3}  {'Name':<30}  {'Nickname':<18}  {'Side'}")
    print(f"{'─'*70}")
    for rec in sorted(records, key=lambda r: r.get("portrait_id", 999)):
        pid  = rec.get("portrait_id", "?")
        name = rec.get("name", "UNKNOWN")
        nick = rec.get("nickname") or ""
        side = rec.get("page_side", "")
        print(f"{pid:>3}  {name:<30}  {nick:<18}  {side}")
    print(f"{'─'*70}")


if __name__ == "__main__":
    main()
