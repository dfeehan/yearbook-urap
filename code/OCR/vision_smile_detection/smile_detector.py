#!/usr/bin/env python3
"""
Smile Detection using LLaVA Vision Model
Analyzes student yearbook photos to detect if they're smiling.
"""

import subprocess
import json
import argparse
import os
import time
from pathlib import Path


def check_ollama():
    """Check if Ollama is installed and LLaVA model is available."""
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            check=True
        )
        if "llava" not in result.stdout:
            print("⚠ LLaVA model not found. Please run: ollama pull llava")
            return False
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("⚠ Ollama is not installed or not running")
        return False


def detect_smile(image_path, model="llava", timeout=300):
    """
    Use LLaVA to detect if the person in the photo is smiling.
    
    Args:
        image_path: Path to the student photo
        model: LLaVA model to use (default: llava)
        timeout: Timeout in seconds (default: 300)
    
    Returns:
        dict with keys: 'smiling' (yes/no/uncertain), 'confidence', 'explanation', 'processing_time'
    """
    start_time = time.time()
    
    try:
        # Construct the full command with image path - explicit English instruction
        cmd = f'ollama run {model} "Is the person in this photo smiling? Answer in English only. Answer: yes, no, or uncertain. Confidence level: high, medium, or low. Brief explanation:" < "{image_path}"'
        
        # Run ollama with the image
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            timeout=timeout,
            text=True
        )
        
        processing_time = time.time() - start_time
        
        if result.returncode != 0:
            return {
                'smiling': 'uncertain',
                'confidence': 'low',
                'explanation': f'Error processing image: {result.stderr}',
                'processing_time_seconds': processing_time
            }
        
        response = result.stdout.strip()
        
        # Parse the response - look for yes/no in the text
        response_lower = response.lower()
        
        smiling = 'uncertain'
        confidence = 'medium'
        
        # Determine if smiling
        if 'smiling' in response_lower or 'smile' in response_lower:
            if any(word in response_lower for word in ['yes', 'is smiling', 'appears to be smiling', 'definitely']):
                smiling = 'yes'
            elif any(word in response_lower for word in ['no', 'not smiling', 'neutral', 'serious']):
                smiling = 'no'
        
        # Check for yes/no at start of response
        if response_lower.startswith('yes'):
            smiling = 'yes'
        elif response_lower.startswith('no'):
            smiling = 'no'
        elif 'uncertain' in response_lower or 'cannot' in response_lower or 'unable' in response_lower:
            smiling = 'uncertain'
        
        # Determine confidence
        if 'high' in response_lower or 'clearly' in response_lower or 'definitely' in response_lower:
            confidence = 'high'
        elif 'low' in response_lower or 'uncertain' in response_lower or 'hard to' in response_lower:
            confidence = 'low'
        else:
            confidence = 'medium'
        
        return {
            'smiling': smiling,
            'confidence': confidence,
            'explanation': response,
            'processing_time_seconds': processing_time
        }
        
    except subprocess.TimeoutExpired:
        return {
            'smiling': 'uncertain',
            'confidence': 'low',
            'explanation': f'Analysis timed out after {timeout} seconds',
            'processing_time_seconds': timeout
        }
    except Exception as e:
        return {
            'smiling': 'uncertain',
            'confidence': 'low',
            'explanation': f'Error: {str(e)}',
            'processing_time_seconds': time.time() - start_time
        }


def process_image(image_path, output_file=None, model="llava", timeout=300):
    """Process a single image and detect smile."""
    print(f"\nAnalyzing: {image_path}")
    
    result = detect_smile(image_path, model, timeout)
    
    # Display results
    print(f"  Smiling: {result['smiling']}")
    print(f"  Confidence: {result['confidence']}")
    print(f"  Explanation: {result['explanation']}")
    print(f"  Processing time: {result['processing_time_seconds']:.2f} seconds")
    
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"  Saved to: {output_file}")
    
    return result


def process_folder(folder_path, output_folder=None, model="llava", timeout=300):
    """Process all images in a folder."""
    folder = Path(folder_path)
    
    if not folder.exists():
        print(f"⚠ Folder not found: {folder_path}")
        return []
    
    # Find all image files
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        image_files.extend(folder.glob(ext))
    
    if not image_files:
        print(f"⚠ No images found in {folder_path}")
        return []
    
    print(f"\nFound {len(image_files)} images to process")
    
    results = []
    
    for i, img_path in enumerate(image_files, 1):
        print(f"\n[{i}/{len(image_files)}]", end=" ")
        
        if output_folder:
            output_path = Path(output_folder) / f"{img_path.stem}_smile.json"
            output_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            output_path = None
        
        result = process_image(str(img_path), output_path, model, timeout)
        result['filename'] = img_path.name
        results.append(result)
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    smiling_count = sum(1 for r in results if r['smiling'] == 'yes')
    not_smiling_count = sum(1 for r in results if r['smiling'] == 'no')
    uncertain_count = sum(1 for r in results if r['smiling'] == 'uncertain')
    
    print(f"Total photos: {len(results)}")
    print(f"Smiling: {smiling_count}")
    print(f"Not smiling: {not_smiling_count}")
    print(f"Uncertain: {uncertain_count}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Detect smiles in yearbook photos using LLaVA vision model'
    )
    parser.add_argument(
        '--input',
        required=True,
        help='Path to image file or folder containing images'
    )
    parser.add_argument(
        '--output',
        help='Path to save results (JSON file for single image, folder for batch)'
    )
    parser.add_argument(
        '--model',
        default='llava',
        help='LLaVA model to use (default: llava)'
    )
    parser.add_argument(
        '--timeout',
        type=int,
        default=300,
        help='Timeout in seconds for each image analysis (default: 300)'
    )
    
    args = parser.parse_args()
    
    # Check Ollama
    print("Checking Ollama installation...")
    if not check_ollama():
        return
    
    print("✓ Ollama is available")
    print(f"✓ LLaVA model is available")
    print(f"\nUsing model: {args.model}")
    print(f"Timeout: {args.timeout} seconds per image")
    
    # Process input
    input_path = Path(args.input)
    
    if input_path.is_file():
        # Single image
        process_image(str(input_path), args.output, args.model, args.timeout)
    elif input_path.is_dir():
        # Folder of images
        process_folder(str(input_path), args.output, args.model, args.timeout)
    else:
        print(f"⚠ Input not found: {args.input}")


if __name__ == "__main__":
    main()
