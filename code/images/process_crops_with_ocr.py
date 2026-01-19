"""
Process yearbook crops using traditional OCR (Tesseract) and LLM correction.
This script fetches annotations from Label Studio, crops the text regions,
runs Tesseract OCR, corrects the output with a local LLM, and saves the results to a CSV.
"""

import os
import sys
import requests
import pandas as pd
import numpy as np
import cv2
import pytesseract
from PIL import Image
from io import BytesIO
from collections import defaultdict
from dotenv import load_dotenv
from pathlib import Path
from tqdm import tqdm

# Add path for LLM correction module
current_dir = Path(__file__).parent
ocr_base = current_dir.parent / 'OCR'
sys.path.append(str(ocr_base / 'ocr_correction'))

try:
    from llm_ocr_correction import correct_text_with_llm
except ImportError:
    print("Warning: Could not import llm_ocr_correction. LLM correction will be skipped.")
    def correct_text_with_llm(text, model="mistral"): return text

# Load environment variables
load_dotenv()

API_TOKEN = os.getenv('LABEL_STUDIO_TOKEN')
TASK_ID = int(os.getenv('TASK_ID', 185777732))
BASE_URL = os.getenv('LABEL_STUDIO_BASE_URL', 'https://app.humansignal.com')
OUTPUT_BASE_DIR = os.getenv('OUTPUT_BASE_DIR', os.getcwd())

if not API_TOKEN:
    raise ValueError("LABEL_STUDIO_TOKEN environment variable is required.")

# ------------------------------------------------------------------
# Helper Functions from OCR_yearbook_pages.py
# ------------------------------------------------------------------
def preprocess_image(image_cv):
    """
    Preprocess image for Tesseract OCR.
    Args:
        image_cv: OpenCV image (numpy array)
    Returns:
        Processed binary image
    """
    # Grayscale
    if len(image_cv.shape) == 3:
        gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
    else:
        gray = image_cv
    
    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # Denoise
    denoised = cv2.bilateralFilter(enhanced, 9, 75, 75)

    # Light blur
    blurred = cv2.GaussianBlur(denoised, (3, 3), 0)
    
    # Upscale 2x
    upscaled = cv2.resize(blurred, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    
    # Adaptive thresholding
    binary = cv2.adaptiveThreshold(
        upscaled, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 11, 2
    )
    
    return binary

def run_tesseract(pil_image):
    """
    Run Tesseract OCR on a PIL image.
    """
    # Convert PIL to OpenCV
    img_cv = np.array(pil_image)
    img_cv = img_cv[:, :, ::-1].copy() # RGB to BGR
    
    # Preprocess
    processed = preprocess_image(img_cv)
    
    # Run Tesseract
    try:
        text = pytesseract.image_to_string(processed)
        return text.strip()
    except pytesseract.TesseractNotFoundError:
        print("\nError: Tesseract is not installed or not in your PATH.")
        print("Please install it using: brew install tesseract (on macOS)")
        print("Or download from: https://github.com/tesseract-ocr/tesseract")
        sys.exit(1)
    except Exception as e:
        print(f"OCR Error: {e}")
        return ""

# ------------------------------------------------------------------
# Helper Functions from URAPCrop.py
# ------------------------------------------------------------------
def fetch_yearbook_annotations():
    response = requests.get(
        f"{BASE_URL}/api/tasks/{TASK_ID}/annotations/",
        headers={"Authorization": f"Token {API_TOKEN}"}
    )
    response.raise_for_status()
    return response.json()

def get_task_image_url():
    response = requests.get(
        f"{BASE_URL}/api/tasks/{TASK_ID}/",
        headers={"Authorization": f"Token {API_TOKEN}"}
    )
    response.raise_for_status()
    task_data = response.json()
    if 'data' in task_data and 'image' in task_data['data']:
        image_path = task_data['data']['image']
        if image_path.startswith('http'):
            return image_path
        else:
            return f"{BASE_URL}{image_path}"
    return None

def organize_by_relations(annotations):
    photo_groups = []
    for annotation in annotations:
        annotation_id = annotation['id']
        created_by = annotation['created_username']
        results = annotation['result']
        
        rectangles = {}
        choices = {}
        texts = {}
        relations = []
        
        for result in results:
            result_type = result.get('type')
            if result_type == 'relation':
                relations.append(result)
            else:
                result_id = result.get('id')
                if not result_id: continue
                
                if result_type == 'rectanglelabels':
                    rectangles[result_id] = result
                elif result_type == 'choices':
                    choices[result_id] = result
                elif result_type == 'textarea':
                    texts[result_id] = result
        
        photo_rects = {rid: r for rid, r in rectangles.items() if r.get('from_name') == 'photo_box'}
        text_rects = {rid: r for rid, r in rectangles.items() if r.get('from_name') == 'text_box'}
        
        relation_graph = defaultdict(list)
        for relation in relations:
            from_id = relation.get('from_id')
            to_id = relation.get('to_id')
            if from_id and to_id:
                relation_graph[from_id].append(to_id)
                relation_graph[to_id].append(from_id)
        
        for photo_id, photo_rect in photo_rects.items():
            visited = set()
            connected_elements = []
            def dfs(element_id):
                if element_id in visited: return
                visited.add(element_id)
                connected_elements.append(element_id)
                for connected_id in relation_graph.get(element_id, []):
                    dfs(connected_id)
            dfs(photo_id)
            
            photo_group = {
                'annotation_id': annotation_id,
                'created_by': created_by,
                'photo_rect': photo_rect,
                'photo_id': photo_id,
                'choices': {},
                'related_texts': [],
                'text_rects': [],
                'connected_ids': connected_elements
            }
            
            for elem_id in connected_elements:
                if elem_id in choices:
                    choice = choices[elem_id]
                    category = choice.get('from_name')
                    value = choice.get('value', {}).get('choices', [])
                    if category and value:
                        photo_group['choices'][category] = value[0] if value else None
                elif elem_id in text_rects:
                    photo_group['text_rects'].append(text_rects[elem_id])
                elif elem_id in texts:
                    photo_group['related_texts'].append(texts[elem_id])
            
            photo_groups.append(photo_group)
    return photo_groups

def crop_region(img, region_data):
    W, H = img.size
    x_perc = region_data.get('x', 0)
    y_perc = region_data.get('y', 0)
    w_perc = region_data.get('width', 0)
    h_perc = region_data.get('height', 0)
    
    left = int((x_perc / 100.0) * W)
    top = int((y_perc / 100.0) * H)
    right = int(left + (w_perc / 100.0) * W)
    bottom = int(top + (h_perc / 100.0) * H)
    
    return img.crop((max(0, left), max(0, top), min(W, right), min(H, bottom)))

# ------------------------------------------------------------------
# Main Processing Logic
# ------------------------------------------------------------------
def main():
    # Check for Tesseract availability
    try:
        pytesseract.get_tesseract_version()
    except pytesseract.TesseractNotFoundError:
        print("Error: Tesseract is not installed. Please install it (e.g., 'brew install tesseract')")
        sys.exit(1)
    except Exception:
        print("Warning: Could not verify Tesseract version. Continuing...")
    
    print("Fetching annotations...")
    annotations = fetch_yearbook_annotations()
    image_url = get_task_image_url()
    
    print("Organizing groups...")
    photo_groups = organize_by_relations(annotations)
    print(f"Found {len(photo_groups)} photo groups.")
    
    print(f"Downloading image from {image_url}...")
    # Use token for image download if needed (copied from URAPCrop.py)
    headers = {"Authorization": f"Token {API_TOKEN}"}
    resp = requests.get(image_url, timeout=30, headers=headers)
    resp.raise_for_status()
    img = Image.open(BytesIO(resp.content)).convert("RGB")
    
    results = []
    
    print("Processing groups with Tesseract OCR + LLM Correction...")
    for i, group in enumerate(tqdm(photo_groups)):
        # Basic metadata
        choices = group['choices']
        
        # Get coordinates
        photo_rect = group['photo_rect']
        val = photo_rect.get('value', {})
        coords = f"x={val.get('x',0):.1f}%, y={val.get('y',0):.1f}%, w={val.get('width',0):.1f}%, h={val.get('height',0):.1f}%"
        
        metadata = {
            'index': i,
            'photo_id': group['photo_id'],
            'source_url': image_url,
            'coordinates': coords,
            'class_year': choices.get('class_year', 'unknown'),
            'gender': choices.get('gender', 'unknown'),
            'race': choices.get('black', 'unknown'),
            'image_quality': choices.get('image_quality', 'unknown'),
        }
        
        # Process text regions
        names_raw = []
        names_corrected = []
        additional_raw = []
        additional_corrected = []
        
        for text_rect in group['text_rects']:
            text_labels = text_rect.get('value', {}).get('rectanglelabels', [])
            label = text_labels[0] if text_labels else 'unknown'
            
            # Crop
            text_crop = crop_region(img, text_rect.get('value', {}))
            
            # OCR
            raw_text = run_tesseract(text_crop)
            
            # LLM Correction
            if raw_text:
                corrected_text = correct_text_with_llm(raw_text)
            else:
                corrected_text = ""
            
            # Store
            if label == 'Name':
                names_raw.append(raw_text)
                names_corrected.append(corrected_text)
            elif label == 'Additional Text':
                additional_raw.append(raw_text)
                additional_corrected.append(corrected_text)
                
        metadata['name_raw'] = '; '.join(names_raw)
        metadata['name_corrected'] = '; '.join(names_corrected)
        metadata['additional_raw'] = '; '.join(additional_raw)
        metadata['additional_corrected'] = '; '.join(additional_corrected)
        
        results.append(metadata)
        
    # Save to CSV
    output_csv = os.path.join(OUTPUT_BASE_DIR, "yearbook_crops", "yearbook_ocr_results.csv")
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"Saved results to {output_csv}")

if __name__ == "__main__":
    main()
