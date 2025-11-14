"""
Direct OCR using LLaVA vision model via Ollama

Uses LLaVA vision model to directly extract text from images without
requiring pre-processed OCR. The vision model can see the image and
extract text more accurately than traditional OCR.

Usage:
    python code/OCR/llava_vision_ocr.py --input image.png --output extracted.txt
    
    # Or process multiple images in a folder
    python code/OCR/llava_vision_ocr.py --input-folder images/ --output-folder output/
    
    # Specify LLaVA model size
    python code/OCR/llava_vision_ocr.py --input image.png --output text.txt --model llava:13b

Prerequisites:
    - Install Ollama: https://ollama.ai
    - Pull LLaVA model: ollama pull llava
    - Make sure ollama is running: ollama serve

Available LLaVA models:
    - llava (default, ~4.7GB) - General purpose, good balance
    - llava:7b (~4.7GB) - Same as llava
    - llava:13b (~7.3GB) - More accurate, slower
    - llava:34b (~19GB) - Most accurate, requires more resources

Example:
    # Start ollama server (in separate terminal if needed)
    ollama serve
    
    # Pull LLaVA model
    ollama pull llava
    
    # Extract text from single image
    python code/OCR/llava_vision_ocr.py --input yearbook_page.png --output text.txt
    
    # Process folder of images
    python code/OCR/llava_vision_ocr.py --input-folder crops/ --output-folder extracted/
"""

import argparse
import subprocess
import sys
import base64
from pathlib import Path
from typing import Optional, List
import json


def check_ollama_available():
    """Check if ollama is installed and accessible."""
    try:
        result = subprocess.run(['ollama', 'list'], 
                              capture_output=True, 
                              text=True, 
                              timeout=5)
        if result.returncode == 0:
            return True, result.stdout
        return False, "Ollama installed but returned error"
    except FileNotFoundError:
        return False, "Ollama not found. Install from https://ollama.ai"
    except Exception as e:
        return False, f"Error checking ollama: {e}"


def check_llava_installed():
    """Check if LLaVA model is installed."""
    try:
        result = subprocess.run(['ollama', 'list'], 
                              capture_output=True, 
                              text=True, 
                              timeout=5)
        if result.returncode == 0:
            return 'llava' in result.stdout.lower()
        return False
    except:
        return False


def encode_image_to_base64(image_path: str) -> str:
    """Encode image file to base64 string."""
    with open(image_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')


def extract_text_with_llava(image_path: str, model: str = "llava") -> Optional[str]:
    """
    Use LLaVA vision model to extract text from an image.
    
    Args:
        image_path: Path to the image file
        model: LLaVA model to use (llava, llava:7b, llava:13b, llava:34b)
        
    Returns:
        Extracted text or None if extraction fails
    """
    try:
        # Prepare the prompt for text extraction
        prompt = """Extract all text from this image exactly as it appears.

Rules:
1. Transcribe ALL visible text precisely
2. Maintain original line breaks and formatting
3. Preserve capitalization exactly
4. Include punctuation marks
5. If text is unclear, make your best judgment based on context
6. Do not add explanations or commentary
7. Return only the extracted text

Extracted text:"""

        # Read and encode the image
        with open(image_path, 'rb') as f:
            image_data = base64.b64encode(f.read()).decode('utf-8')
        
        # Prepare the API payload for ollama
        payload = {
            "model": model,
            "prompt": prompt,
            "images": [image_data],
            "stream": False
        }
        
        # Call ollama API
        result = subprocess.run(
            ['ollama', 'run', model],
            input=json.dumps(payload),
            capture_output=True,
            text=True,
            timeout=300
        )
        
        if result.returncode == 0:
            # Try to parse as JSON first (API response format)
            try:
                response_data = json.loads(result.stdout)
                if 'response' in response_data:
                    return response_data['response'].strip()
            except json.JSONDecodeError:
                # Fallback to treating as plain text
                pass
            
            # Return stdout as is
            return result.stdout.strip()
        else:
            print(f"⚠ LLaVA extraction failed for {image_path}: {result.stderr}")
            return None
            
    except subprocess.TimeoutExpired:
        print(f"⚠ Timeout processing {image_path}")
        return None
    except Exception as e:
        print(f"⚠ Error processing {image_path}: {e}")
        return None


def extract_text_with_llava_cli(image_path: str, model: str = "llava") -> Optional[str]:
    """
    Alternative method using ollama CLI with image path directly.
    
    Args:
        image_path: Path to the image file
        model: LLaVA model to use
        
    Returns:
        Extracted text or None if extraction fails
    """
    try:
        # Prepare the prompt
        prompt = """Extract all text from this image exactly as it appears. Maintain original formatting, line breaks, and capitalization. Return only the extracted text without explanations."""
        
        # Use ollama CLI with image file
        # Note: Syntax varies by ollama version, trying the most common format
        cmd = ['ollama', 'run', model, prompt, image_path]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300
        )
        
        if result.returncode == 0:
            return result.stdout.strip()
        else:
            print(f"⚠ CLI extraction failed for {image_path}: {result.stderr}")
            return None
            
    except subprocess.TimeoutExpired:
        print(f"⚠ Timeout processing {image_path}")
        return None
    except Exception as e:
        print(f"⚠ Error processing {image_path}: {e}")
        return None


def process_single_image(image_path: str, output_path: str, model: str = "llava"):
    """Process a single image and save extracted text."""
    print(f"\nProcessing: {image_path}")
    
    # Try CLI method first (simpler and often works better)
    extracted_text = extract_text_with_llava_cli(image_path, model)
    
    if not extracted_text:
        print("  Trying alternative method...")
        extracted_text = extract_text_with_llava(image_path, model)
    
    if extracted_text:
        # Save to output file
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(extracted_text)
        
        print(f"✓ Extracted {len(extracted_text)} characters")
        print(f"  Saved to: {output_path}")
        return True
    else:
        print(f"✗ Failed to extract text")
        return False


def process_folder(input_folder: str, output_folder: str, model: str = "llava"):
    """Process all images in a folder."""
    input_path = Path(input_folder)
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Supported image extensions
    image_extensions = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp', '.gif'}
    
    # Find all images
    image_files = []
    for ext in image_extensions:
        image_files.extend(input_path.glob(f'*{ext}'))
        image_files.extend(input_path.glob(f'*{ext.upper()}'))
    
    if not image_files:
        print(f"No images found in {input_folder}")
        return
    
    print(f"\nFound {len(image_files)} images to process")
    
    success_count = 0
    for i, img_file in enumerate(sorted(image_files), 1):
        print(f"\n[{i}/{len(image_files)}]", end=" ")
        
        # Create output filename
        output_file = output_path / f"{img_file.stem}.txt"
        
        if process_single_image(str(img_file), str(output_file), model):
            success_count += 1
    
    print(f"\n{'='*60}")
    print(f"Processing complete!")
    print(f"Successfully processed: {success_count}/{len(image_files)} images")
    print(f"Output saved to: {output_folder}")


def main():
    parser = argparse.ArgumentParser(
        description='Extract text from images using LLaVA vision model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single image
  python code/OCR/llava_vision_ocr.py --input page.png --output text.txt
  
  # Folder of images
  python code/OCR/llava_vision_ocr.py --input-folder crops/ --output-folder extracted/
  
  # Use larger model for better accuracy
  python code/OCR/llava_vision_ocr.py --input page.png --output text.txt --model llava:13b
        """
    )
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--input', help='Input image file')
    group.add_argument('--input-folder', help='Input folder containing images')
    
    parser.add_argument('--output', help='Output text file (for single image)')
    parser.add_argument('--output-folder', help='Output folder (for folder processing)')
    
    parser.add_argument('-m', '--model', default='llava',
                       choices=['llava', 'llava:7b', 'llava:13b', 'llava:34b'],
                       help='LLaVA model to use (default: llava)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.input and not args.output:
        print("ERROR: --output required when using --input")
        sys.exit(1)
    
    if args.input_folder and not args.output_folder:
        print("ERROR: --output-folder required when using --input-folder")
        sys.exit(1)
    
    # Check ollama availability
    print("Checking Ollama installation...")
    available, message = check_ollama_available()
    if not available:
        print(f"ERROR: {message}")
        print("\nTo install Ollama:")
        print("1. Visit https://ollama.ai")
        print("2. Download and install for your OS")
        print("3. Run: ollama pull llava")
        print("4. Start server: ollama serve")
        sys.exit(1)
    
    print(f"✓ Ollama is available")
    
    # Check if LLaVA is installed
    if not check_llava_installed():
        print(f"\n⚠ LLaVA model not found")
        print(f"Installing LLaVA model: {args.model}")
        print("This may take a few minutes...")
        
        result = subprocess.run(['ollama', 'pull', args.model], 
                              capture_output=True, 
                              text=True)
        if result.returncode != 0:
            print(f"ERROR: Failed to install {args.model}")
            print(result.stderr)
            sys.exit(1)
        print(f"✓ {args.model} installed successfully")
    else:
        print(f"✓ LLaVA model is available")
    
    print(f"\nUsing model: {args.model}")
    
    # Process images
    if args.input:
        # Single image
        if not Path(args.input).exists():
            print(f"ERROR: Input file not found: {args.input}")
            sys.exit(1)
        
        success = process_single_image(args.input, args.output, args.model)
        sys.exit(0 if success else 1)
    else:
        # Folder of images
        if not Path(args.input_folder).exists():
            print(f"ERROR: Input folder not found: {args.input_folder}")
            sys.exit(1)
        
        process_folder(args.input_folder, args.output_folder, args.model)


if __name__ == '__main__':
    main()
