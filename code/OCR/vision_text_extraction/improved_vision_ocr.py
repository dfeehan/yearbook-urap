"""
Improved OCR using advanced vision models via ollama-ocr package

Uses state-of-the-art vision models (llama3.2-vision, granite3.2-vision, minicpm-v)
for accurate text extraction from historical yearbook images.

This implementation uses the ollama-ocr package which provides:
- Better accuracy than basic LLaVA
- Multiple output formats (markdown, text, json, structured)
- Batch processing support
- Language specification

Usage:
    # Single image
    python improved_vision_ocr.py --input image.jpg --output text.txt
    
    # Batch processing
    python improved_vision_ocr.py --input-folder images/ --output-folder output/
    
    # Specify model and format
    python improved_vision_ocr.py --input image.jpg --model llama3.2-vision:11b --format markdown

Prerequisites:
    - Install: pip install ollama-ocr
    - Pull model: ollama pull llama3.2-vision:11b
    - Or: ollama pull granite3.2-vision
    - Or: ollama pull minicpm-v
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, Dict, List
import json
import time
from ollama_ocr import OCRProcessor


def process_single_image(
    image_path: str,
    output_path: Optional[str] = None,
    model: str = "llama3.2-vision:11b",
    format_type: str = "text",
    custom_prompt: Optional[str] = None,
    language: str = "English"
) -> str:
    """
    Process a single image and extract text
    
    Args:
        image_path: Path to input image
        output_path: Optional path to save output
        model: Vision model to use (llama3.2-vision:11b, granite3.2-vision, minicpm-v)
        format_type: Output format (markdown, text, json, structured, key_value, table)
        custom_prompt: Optional custom extraction instructions
        language: Language of text in image
    
    Returns:
        Extracted text
    """
    print(f"Processing: {image_path}")
    print(f"Model: {model}")
    print(f"Format: {format_type}")
    
    # Initialize OCR processor
    ocr = OCRProcessor(model_name=model)
    
    # Default prompt for yearbook pages if none provided
    # STRICT extraction prompt to prevent summarization/hallucination
    if custom_prompt is None:
        custom_prompt = (
            "Output ONLY the exact textual content present in this image. "
            "Do NOT summarize, paraphrase, describe, or add any commentary. "
            "Preserve line breaks, spacing, and punctuation exactly as shown. "
            "Output raw text only - no headings, explanations, or extra words."
        )
    
    # Process image with runtime tracking
    start_time = time.time()
    result = ocr.process_image(
        image_path=image_path,
        format_type=format_type,
        custom_prompt=custom_prompt,
        language=language
    )
    processing_time = time.time() - start_time
    
    print(f"Processing time: {processing_time:.2f} seconds")
    
    # Save output if path provided
    if output_path:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            if format_type == 'json':
                json.dump(result, f, indent=2, ensure_ascii=False)
            else:
                f.write(str(result))
        
        print(f"✓ Saved output to: {output_path}")
    
    return result


def process_batch(
    input_folder: str,
    output_folder: str,
    model: str = "llama3.2-vision:11b",
    format_type: str = "text",
    custom_prompt: Optional[str] = None,
    language: str = "English",
    max_workers: int = 4,
    recursive: bool = False
) -> Dict:
    """
    Process multiple images in batch
    
    Args:
        input_folder: Directory containing images
        output_folder: Directory to save outputs
        model: Vision model to use
        format_type: Output format
        custom_prompt: Optional custom extraction instructions
        language: Language of text in images
        max_workers: Number of parallel workers
        recursive: Search subdirectories
    
    Returns:
        Dictionary with results and statistics
    """
    print(f"Batch processing folder: {input_folder}")
    print(f"Model: {model}")
    print(f"Max workers: {max_workers}")
    
    # Initialize OCR processor with parallel processing
    ocr = OCRProcessor(model_name=model, max_workers=max_workers)
    
    # Default prompt for yearbook pages - STRICT extraction
    if custom_prompt is None:
        custom_prompt = (
            "Output ONLY the exact textual content present in this image. "
            "Do NOT summarize, paraphrase, describe, or add any commentary. "
            "Preserve line breaks, spacing, and punctuation exactly as shown. "
            "Output raw text only - no headings, explanations, or extra words."
        )
    
    # Process batch with runtime tracking
    start_time = time.time()
    batch_results = ocr.process_batch(
        input_path=input_folder,
        format_type=format_type,
        recursive=recursive,
        preprocess=True,  # Enable image preprocessing
        custom_prompt=custom_prompt,
        language=language
    )
    total_time = time.time() - start_time
    
    # Save individual results
    output_dir = Path(output_folder)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for file_path, text in batch_results['results'].items():
        # Create output filename
        input_file = Path(file_path)
        output_file = output_dir / f"{input_file.stem}_extracted.txt"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            if format_type == 'json':
                json.dump(text, f, indent=2, ensure_ascii=False)
            else:
                f.write(str(text))
        
        print(f"✓ Processed: {input_file.name} -> {output_file.name}")
    
    # Print statistics
    stats = batch_results['statistics']
    print("\n" + "="*60)
    print("Processing Statistics:")
    print(f"  Total images: {stats['total']}")
    print(f"  Successfully processed: {stats['successful']}")
    print(f"  Failed: {stats['failed']}")
    print(f"  Total time: {total_time:.2f} seconds")
    if stats['successful'] > 0:
        print(f"  Average time per image: {total_time/stats['successful']:.2f} seconds")
    print("="*60)
    
    return batch_results


def main():
    parser = argparse.ArgumentParser(
        description="Extract text from images using advanced vision models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single image with llama3.2-vision
  python improved_vision_ocr.py --input yearbook_page.jpg --output text.txt
  
  # Single image with granite3.2-vision (better for documents)
  python improved_vision_ocr.py --input page.jpg --model granite3.2-vision --format markdown
  
  # Batch processing
  python improved_vision_ocr.py --input-folder images/ --output-folder results/
  
  # Custom prompt
  python improved_vision_ocr.py --input page.jpg --prompt "Extract only student names"

Recommended Models:
  - llama3.2-vision:11b (default) - Best general purpose
  - granite3.2-vision - Specialized for documents/tables
  - minicpm-v - High resolution support (1.8M pixels)
  
Output Formats:
  - text (default) - Plain text
  - markdown - Preserves formatting with headers/lists
  - json - Structured data
  - structured - Tables and organized data
  - key_value - Labeled information
  - table - Extract tabular data
        """
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--input', help='Input image file')
    input_group.add_argument('--input-folder', help='Input folder containing images')
    
    # Output options
    parser.add_argument('--output', help='Output file (for single image)')
    parser.add_argument('--output-folder', help='Output folder (for batch processing)')
    
    # Model and format options
    parser.add_argument(
        '--model',
        default='llama3.2-vision:11b',
        help='Vision model (llama3.2-vision:11b, granite3.2-vision, minicpm-v)'
    )
    parser.add_argument(
        '--format',
        default='text',
        choices=['text', 'markdown', 'json', 'structured', 'key_value', 'table'],
        help='Output format'
    )
    parser.add_argument('--prompt', help='Custom extraction prompt')
    parser.add_argument('--language', default='English', help='Language of text')
    
    # Batch processing options
    parser.add_argument('--workers', type=int, default=4, help='Parallel workers for batch')
    parser.add_argument('--recursive', action='store_true', help='Search subdirectories')
    
    args = parser.parse_args()
    
    try:
        if args.input:
            # Single image processing
            if not Path(args.input).exists():
                print(f"Error: Input file not found: {args.input}")
                sys.exit(1)
            
            result = process_single_image(
                image_path=args.input,
                output_path=args.output,
                model=args.model,
                format_type=args.format,
                custom_prompt=args.prompt,
                language=args.language
            )
            
            # Print result if no output file specified
            if not args.output:
                print("\n" + "="*60)
                print("Extracted Text:")
                print("="*60)
                print(result)
                print("="*60)
        
        else:
            # Batch processing
            if not Path(args.input_folder).exists():
                print(f"Error: Input folder not found: {args.input_folder}")
                sys.exit(1)
            
            if not args.output_folder:
                args.output_folder = str(Path(args.input_folder) / 'output')
                print(f"No output folder specified, using: {args.output_folder}")
            
            process_batch(
                input_folder=args.input_folder,
                output_folder=args.output_folder,
                model=args.model,
                format_type=args.format,
                custom_prompt=args.prompt,
                language=args.language,
                max_workers=args.workers,
                recursive=args.recursive
            )
    
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
