#Install Tesseract--for mac write "brew install tesseract" in terminal
import pytesseract #pip install pytesseract in terminal
import cv2 #pip install opencv
from PIL import Image
import sys

image_path = '185777725.jpg'
img_cv = cv2.imread(image_path)

h, w = img_cv.shape[:2]
mid = w // 2
#left = img_cv[:, :mid]
right = img_cv[:, mid:]

def preprocess(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    # This enhances local contrast without affecting overall brightness
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # Denoise using bilateral filter (preserves edges)
    denoised = cv2.bilateralFilter(enhanced, 9, 75, 75)
    
    # Apply slight blur to reduce noise while keeping text sharp
    blurred = cv2.GaussianBlur(denoised, (3, 3), 0)
    
    # Upscale 2x for better OCR accuracy
    upscaled = cv2.resize(blurred, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    
    # Final adaptive thresholding (better than simple threshold for varying lighting)
    # This is more forgiving than OTSU for old documents
    binary = cv2.adaptiveThreshold(
        upscaled, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 11, 2
    )
    
    cv2.imwrite(f'processed_{image_path}', binary)
    return binary

right_proc = preprocess(right)
text_right = pytesseract.image_to_string(right_proc)
print(text_right)

with open("processed_" + image_path + ".txt", 'w', encoding='utf-8') as f:
    f.write(text_right) #write in txt file


# img = Image.open(image_path)
# text = pytesseract.image_to_string(img)

# print(text) #print to console

# with open(image_path + ".txt", 'w', encoding='utf-8') as f:
#             f.write(text) #write in txt file