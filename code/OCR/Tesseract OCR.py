#Install Tesseract--for mac write "brew install tesseract" in terminal
import pytesseract #pip install pytesseract in terminal
from PIL import Image
import sys

image_path = '185777725.jpg'
img = Image.open(image_path)
text = pytesseract.image_to_string(img)

print(text)

with open(image_path + ".txt", 'w', encoding='utf-8') as f:
            f.write(text)