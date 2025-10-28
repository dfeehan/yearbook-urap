#Install Tesseract--for mac write "brew install tesseract" in terminal
import pytesseract #pip install pytesseract in terminal
import cv2 #pip install opencv
from PIL import Image
import sys
import os
import glob


script_dir = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(script_dir, 'relationship_views_input')
image_files = glob.glob(os.path.join(image_dir, '*.png'))

def preprocess(image):
    #Grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    #CLAHE (Contrast Limited Adaptive Histogram Equalization)
    #Enhances local contrast without affecting overall brightness
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    #Denoise using bilateral filter (preserves edges)
    denoised = cv2.bilateralFilter(enhanced, 9, 75, 75)

    #Light blur to reduce noise while keeping text sharp
    blurred = cv2.GaussianBlur(denoised, (3, 3), 0)
    
    #Upscale 2x for better OCR accuracy
    upscaled = cv2.resize(blurred, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    
    #Final adaptive thresholding (better than simple threshold for varying lighting)
    binary = cv2.adaptiveThreshold(
        upscaled, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 11, 2
    )
    
    return binary

script_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(script_dir, 'relationship_views_output')
os.makedirs(output_dir, exist_ok=True)

for image_path in image_files:
    
    img_cv = cv2.imread(image_path)

    img_proc = preprocess(img_cv)
    text = pytesseract.image_to_string(img_proc)
    
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    
    with open(os.path.join(output_dir, f"processed_{base_name}.txt"), 'w', encoding='utf-8') as f:
        f.write(text)
    
