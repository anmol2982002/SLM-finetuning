#!/usr/bin/env python3
"""
Text Extraction Script using Apache Tika
=========================================
Extracts text from documents in a directory structure where each subdirectory
represents a class/label.

Directory structure expected:
    data/
    ├── OfferLetter/
    │   ├── doc1.pdf
    │   └── doc2.docx
    ├── SalaryStructure/
    │   └── file1.pdf
    └── ...

Usage:
    python scripts/extract_text.py --input_dir data/ --output extracted.jsonl

Requirements:
    - Java 8+ installed (required by Tika)
    - pip install tika tqdm
"""

import os
import sys
import json
import argparse
import logging
import re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def init_tika():
    """Initialize Tika parser with error handling."""
    try:
        from tika import parser
        # Trigger Tika server initialization
        logger.info("Initializing Apache Tika...")
        return parser
    except ImportError:
        logger.error("Tika not installed. Run: pip install tika")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to initialize Tika: {e}")
        logger.error("Make sure Java 8+ is installed and JAVA_HOME is set")
        sys.exit(1)


def clean_text(text: str) -> str:
    """Clean and normalize extracted text."""
    if not text:
        return ""
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove control characters except newlines
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text


def extract_text_from_file(parser, file_path: str) -> Tuple[str, Dict]:
    """
    Extract text from a single file using Apache Tika.
    
    Args:
        parser: Tika parser module
        file_path: Path to the file
        
    Returns:
        Tuple of (extracted_text, metadata_dict)
    """
    try:
        parsed = parser.from_file(file_path)
        text = parsed.get('content') or ""
        metadata = parsed.get('metadata') or {}
        
        # Clean the text
        text = clean_text(text)
        
        return text, {
            'content_type': metadata.get('Content-Type', 'unknown'),
            'pages': metadata.get('xmpTPg:NPages', metadata.get('Page-Count', 'N/A')),
            'success': True
        }
        
    except Exception as e:
        logger.warning(f"Error extracting from {file_path}: {e}")
        return "", {'success': False, 'error': str(e)}


def get_files_by_label(input_dir: str) -> Dict[str, List[str]]:
    """
    Scan directory and return files grouped by label (subdirectory name).
    
    Args:
        input_dir: Root directory containing label subdirectories
        
    Returns:
        Dict mapping label names to list of file paths
    """
    files_by_label = {}
    input_path = Path(input_dir)
    
    if not input_path.exists():
        logger.error(f"Input directory does not exist: {input_dir}")
        sys.exit(1)
    
    for label_dir in input_path.iterdir():
        if not label_dir.is_dir():
            continue
        
        # Skip hidden directories
        if label_dir.name.startswith('.'):
            continue
            
        label = label_dir.name
        files_by_label[label] = []
        
        for file_path in label_dir.iterdir():
            if file_path.is_file() and not file_path.name.startswith('.'):
                files_by_label[label].append(str(file_path))
    
    return files_by_label


def process_directory(
    input_dir: str, 
    output_file: str,
    min_text_length: int = 50,
    max_workers: int = 4,
    validate_only: bool = False
) -> Dict:
    """
    Process all files in the directory structure and extract text.
    
    Args:
        input_dir: Root directory containing label subdirectories
        output_file: Path to output JSONL file
        min_text_length: Minimum text length to include (skip nearly empty extractions)
        max_workers: Number of parallel workers for extraction
        validate_only: If True, only validate structure without extraction
        
    Returns:
        Statistics dictionary
    """
    files_by_label = get_files_by_label(input_dir)
    
    if not files_by_label:
        logger.error(f"No label directories found in {input_dir}")
        sys.exit(1)
    
    # Statistics
    stats = {
        'labels': list(files_by_label.keys()),
        'total_labels': len(files_by_label),
        'files_per_label': {label: len(files) for label, files in files_by_label.items()},
        'total_files': sum(len(files) for files in files_by_label.values()),
        'successful_extractions': 0,
        'failed_extractions': 0,
        'skipped_short_text': 0
    }
    
    logger.info(f"Found {stats['total_labels']} labels with {stats['total_files']} total files")
    
    if validate_only:
        logger.info("Validation mode - skipping extraction")
        print("\n=== Directory Validation ===")
        print(f"Labels found: {stats['total_labels']}")
        for label, count in sorted(stats['files_per_label'].items()):
            print(f"  {label}: {count} files")
        return stats
    
    # Initialize Tika
    parser = init_tika()
    
    # Prepare all file tasks
    all_tasks = []
    for label, files in files_by_label.items():
        for file_path in files:
            all_tasks.append((label, file_path))
    
    # Process files
    results = []
    logger.info(f"Extracting text from {len(all_tasks)} files...")
    
    with tqdm(total=len(all_tasks), desc="Extracting") as pbar:
        # Note: Using single thread for Tika stability, but structure allows parallelization
        for label, file_path in all_tasks:
            text, metadata = extract_text_from_file(parser, file_path)
            
            if metadata['success']:
                if len(text) >= min_text_length:
                    results.append({
                        'label': label,
                        'text': text,
                        'source': file_path,
                        'metadata': metadata
                    })
                    stats['successful_extractions'] += 1
                else:
                    stats['skipped_short_text'] += 1
                    logger.debug(f"Skipped {file_path}: text too short ({len(text)} chars)")
            else:
                stats['failed_extractions'] += 1
            
            pbar.update(1)
    
    # Write output
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in results:
            # Don't include metadata in output to keep it clean
            output_item = {
                'label': item['label'],
                'text': item['text'],
                'source': item['source']
            }
            f.write(json.dumps(output_item, ensure_ascii=False) + '\n')
    
    logger.info(f"Wrote {len(results)} records to {output_file}")
    
    # Print summary
    print("\n" + "=" * 50)
    print("EXTRACTION SUMMARY")
    print("=" * 50)
    print(f"Total labels: {stats['total_labels']}")
    print(f"Total files processed: {stats['total_files']}")
    print(f"Successful extractions: {stats['successful_extractions']}")
    print(f"Failed extractions: {stats['failed_extractions']}")
    print(f"Skipped (too short): {stats['skipped_short_text']}")
    print(f"Output file: {output_file}")
    print("\nFiles per label:")
    for label, count in sorted(stats['files_per_label'].items()):
        print(f"  {label}: {count}")
    print("=" * 50)
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description='Extract text from documents using Apache Tika',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic extraction
  python scripts/extract_text.py --input_dir data/ --output extracted.jsonl
  
  # Validate directory structure only
  python scripts/extract_text.py --input_dir data/ --output extracted.jsonl --validate-only
  
  # With custom minimum text length
  python scripts/extract_text.py --input_dir data/ --output extracted.jsonl --min-length 100
        """
    )
    
    parser.add_argument(
        '--input_dir', '-i',
        required=True,
        help='Input directory containing label subdirectories'
    )
    parser.add_argument(
        '--output', '-o',
        required=True,
        help='Output JSONL file path'
    )
    parser.add_argument(
        '--min-length',
        type=int,
        default=50,
        help='Minimum text length to include (default: 50)'
    )
    parser.add_argument(
        '--validate-only',
        action='store_true',
        help='Only validate directory structure, do not extract'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    process_directory(
        input_dir=args.input_dir,
        output_file=args.output,
        min_text_length=args.min_length,
        validate_only=args.validate_only
    )


if __name__ == '__main__':
    main()
