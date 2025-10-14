import requests
from io import BytesIO
from PIL import Image, ImageDraw # type: ignore
import os

# ------------------------------------------------------------------
# 1. Fake JSON_MIN-like data (single item). We put two rectangles.
#    Coordinates are percentages of the full image dimensions.
# ------------------------------------------------------------------
data = {
    "image": "https://images.unsplash.com/photo-1503023345310-bd7c1de61c7d",  # Random Unsplash portrait-style photo
    "tag": [
        {
            # Fake rectangle #1 (e.g., upper-central region)
            "x": 40.0,          # left % of image width
            "y": 10.0,          # top % of image height
            "width": 20.0,      # box width in %
            "height": 25.0,     # box height in %
            "rotation": 0,
            "rectanglelabels": ["Face"]
        },
        {
            # Fake rectangle #2 (e.g., lower-right region)
            "x": 55.0,
            "y": 55.0,
            "width": 30.0,
            "height": 35.0,
            "rotation": 0,
            "rectanglelabels": ["Body"]
        }
    ]
}

# Output directories - save to desktop
DESKTOP_PATH = "/Users/louis/Desktop"
CROP_DIR = os.path.join(DESKTOP_PATH, "crops")
os.makedirs(CROP_DIR, exist_ok=True)

# ------------------------------------------------------------------
# 2. Load the image
# ------------------------------------------------------------------
resp = requests.get(data["image"], timeout=30)
resp.raise_for_status()
img = Image.open(BytesIO(resp.content)).convert("RGB")
W, H = img.size
print(f"Loaded image size: {W} x {H}")

# ------------------------------------------------------------------
# 3. Iterate annotations and crop
# ------------------------------------------------------------------
for i, ann in enumerate(data["tag"]):
    x_perc = ann["x"]
    y_perc = ann["y"]
    w_perc = ann["width"]
    h_perc = ann["height"]
    rot = ann.get("rotation", 0)
    labels = ann.get("rectanglelabels", [])

    # Convert percentages to pixel coordinates
    left   = (x_perc / 100.0) * W
    top    = (y_perc / 100.0) * H
    right  = left + (w_perc / 100.0) * W
    bottom = top  + (h_perc / 100.0) * H

    # Clamp and convert to int
    left_i   = max(0, int(round(left)))
    top_i    = max(0, int(round(top)))
    right_i  = min(W, int(round(right)))
    bottom_i = min(H, int(round(bottom)))

    # Crop
    crop = img.crop((left_i, top_i, right_i, bottom_i))

    # Simple rotation handling (if rotation != 0)
    if rot and rot % 360 != 0:
        crop = crop.rotate(-rot, expand=True)

    label_part = "_".join(labels) if labels else "nolabel"
    out_name = f"crop_{i}_{label_part}.png"
    crop.save(os.path.join(CROP_DIR, out_name))
    print(f"Saved {out_name}: ({left_i}, {top_i}, {right_i}, {bottom_i})")

# ------------------------------------------------------------------
# 4. (Optional) Visual confirmation: draw rectangles & save overlay
# ------------------------------------------------------------------
overlay = img.copy()
draw = ImageDraw.Draw(overlay)
for ann in data["tag"]:
    left = int(round((ann["x"] / 100.0) * W))
    top = int(round((ann["y"] / 100.0) * H))
    right = int(round(left + (ann["width"] / 100.0) * W))
    bottom = int(round(top + (ann["height"] / 100.0) * H))
    draw.rectangle([left, top, right, bottom], outline="red", width=4)
    label = ",".join(ann.get("rectanglelabels", []))
    if label:
        draw.text((left + 3, top + 3), label, fill="red")

overlay_path = os.path.join(DESKTOP_PATH, "overlay_with_boxes.png")
overlay.save(overlay_path)
print(f"Saved overlay image with drawn boxes -> {overlay_path}")