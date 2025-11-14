"""
Simple LLM-based OCR correction using local Ollama

Takes OCR text output and improves it using context-aware LLM correction.
No complex glossaries - just leverages the LLM's ability to understand context
and fix common OCR errors (O/0, l/1, rn/m, etc.)

Usage:
    python code/OCR/llm_ocr_correction.py --input ocr_output.txt --output corrected.txt
    
    # Or specify a different model
    python code/OCR/llm_ocr_correction.py --input ocr_output.txt --output corrected.txt --model llama3.1

Prerequisites:
    - Install Ollama: https://ollama.ai
    - Pull a model: ollama pull mistral (or llama3.1, qwen2.5, etc.)
    - Make sure ollama is running: ollama serve

Example:
    # Start ollama server (in separate terminal)
    ollama serve
    
    # Pull a model if you haven't
    ollama pull mistral
    
    # Run correction
    python code/OCR/llm_ocr_correction.py --input my_ocr.txt --output fixed.txt
"""

import argparse
import subprocess
import sys
from pathlib import Path


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


def correct_text_with_llm(text: str, model: str = "mistral", chunk_lines: int = 5) -> str:
    """
    Send OCR text to local LLM for context-aware correction.
    
    Args:
        text: Raw OCR text to correct
        model: Ollama model name (default: mistral)
        chunk_lines: Process this many lines at a time for better context
        
    Returns:
        Corrected text
    """
    lines = text.split('\n')  # Don't strip - preserve blank lines
    corrected_lines = []
    
    # Process in chunks, but preserve empty lines
    i = 0
    while i < len(lines):
        # Collect non-empty lines for this chunk
        chunk = []
        chunk_start_idx = i
        
        while len(chunk) < chunk_lines and i < len(lines):
            if lines[i].strip():  # Non-empty line
                chunk.append(lines[i])
            else:  # Empty line - preserve it
                if chunk:  # If we have content, process it first
                    break
                else:  # Empty line at chunk start - just keep it
                    corrected_lines.append('')
                    i += 1
                    continue
            i += 1
        
        if not chunk:  # No more content
            break
            
        chunk_text = '\n'.join(chunk)
        
        # Build a focused prompt that leverages LLM's context understanding
        prompt = f"""You are correcting OCR errors in historical yearbook text.

CRITICAL: You MUST preserve the exact line structure. Return the SAME number of lines.

Common OCR mistakes to fix:
- O/0 confusion (e.g., "0liver" → "Oliver")
- l/I/1 confusion (e.g., "He1lo" → "Hello")  
- rn/m confusion (e.g., "narne" → "name")
- cl/d confusion (e.g., "clon't" → "don't")
- Punctuation errors

Rules:
1. Fix ONLY obvious OCR character errors
2. Keep each line separate - DO NOT merge or split lines
3. Preserve capitalization patterns (all-caps names stay all-caps)
4. If unsure, keep the original
5. Return EXACTLY {len(chunk)} lines (one per line in input)

Input has {len(chunk)} lines. Output MUST have {len(chunk)} lines.

OCR Text:
{chunk_text}

Corrected Text (return ONLY the corrected text with same line structure):"""

        try:
            # Call ollama with the prompt
            result = subprocess.run(
                ['ollama', 'run', model],
                input=prompt,
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode == 0:
                corrected = result.stdout.strip()
                # Remove any extra commentary the LLM might add
                if corrected:
                    # Split corrected output back into lines
                    corrected_chunk_lines = corrected.split('\n')
                    corrected_lines.extend(corrected_chunk_lines)
                    print(f"✓ Corrected chunk (lines {chunk_start_idx+1}-{i})")
                else:
                    corrected_lines.extend(chunk)
                    print(f"⚠ Chunk (lines {chunk_start_idx+1}-{i}): No output, keeping original")
            else:
                print(f"⚠ Chunk (lines {chunk_start_idx+1}-{i}): Error, keeping original")
                corrected_lines.extend(chunk)
                
        except subprocess.TimeoutExpired:
            print(f"⚠ Chunk (lines {chunk_start_idx+1}-{i}): Timeout, keeping original")
            corrected_lines.extend(chunk)
        except Exception as e:
            print(f"⚠ Chunk (lines {chunk_start_idx+1}-{i}): Error ({e}), keeping original")
            corrected_lines.extend(chunk)
    
    return '\n'.join(corrected_lines)


def main():
    parser = argparse.ArgumentParser(
        description='Improve OCR text using local LLM context-aware correction',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python code/OCR/llm_ocr_correction.py --input ocr.txt --output fixed.txt
  python code/OCR/llm_ocr_correction.py -i ocr.txt -o fixed.txt --model llama3.1
  python code/OCR/llm_ocr_correction.py -i ocr.txt -o fixed.txt --chunk-size 20
        """
    )
    
    parser.add_argument('-i', '--input', required=True, 
                       help='Input file with OCR text')
    parser.add_argument('-o', '--output', required=True,
                       help='Output file for corrected text')
    parser.add_argument('-m', '--model', default='mistral',
                       help='Ollama model to use (default: mistral)')
    parser.add_argument('--chunk-size', type=int, default=5,
                       help='Number of lines to process at once (default: 5)')
    parser.add_argument('--show-diff', action='store_true',
                       help='Show differences between original and corrected')
    
    args = parser.parse_args()
    
    # Check if input file exists
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: Input file not found: {args.input}")
        sys.exit(1)
    
    # Check ollama availability
    print("Checking Ollama installation...")
    available, message = check_ollama_available()
    if not available:
        print(f"ERROR: {message}")
        print("\nTo install Ollama:")
        print("1. Visit https://ollama.ai")
        print("2. Download and install for your OS")
        print("3. Run: ollama pull mistral")
        print("4. Start server: ollama serve")
        sys.exit(1)
    
    print(f"✓ Ollama is available")
    print(f"\nAvailable models:\n{message}")
    
    # Read input
    print(f"\nReading OCR text from: {args.input}")
    with open(args.input, 'r', encoding='utf-8') as f:
        original_text = f.read()
    
    print(f"Original text: {len(original_text)} characters, {len(original_text.splitlines())} lines")
    
    # Correct with LLM
    print(f"\nCorrecting with {args.model} (chunk size: {args.chunk_size} lines)...")
    corrected_text = correct_text_with_llm(original_text, args.model, args.chunk_size)
    
    # Write output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        f.write(corrected_text)
    
    print(f"\n✓ Corrected text saved to: {args.output}")
    print(f"Corrected text: {len(corrected_text)} characters, {len(corrected_text.splitlines())} lines")
    
    # Show diff if requested
    if args.show_diff:
        print("\n" + "="*60)
        print("COMPARISON (Original → Corrected)")
        print("="*60)
        orig_lines = original_text.splitlines()
        corr_lines = corrected_text.splitlines()
        
        for i, (orig, corr) in enumerate(zip(orig_lines, corr_lines), 1):
            if orig != corr:
                print(f"\nLine {i}:")
                print(f"  Original:  {orig}")
                print(f"  Corrected: {corr}")


if __name__ == '__main__':
    main()
