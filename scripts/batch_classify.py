#!/usr/bin/env python3
"""
Batch Document Classification Script
=====================================
Classify documents from a directory structure where subdirectory names represent labels.

Usage:
    python scripts/batch_classify.py --input_dir ./test_data --output_file ./results.json

Directory Structure Expected:
    input_dir/
    ├── LabelA/
    │   ├── doc1.txt
    │   └── doc2.txt
    ├── LabelB/
    │   ├── file1.txt
    │   └── file2.txt
    └── ...
"""

import argparse
import json
import os
import sys
import logging
from pathlib import Path
from typing import Dict, List, Optional
from collections import defaultdict
import csv
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def find_text_files(directory: Path) -> Dict[str, List[Path]]:
    """
    Find all text files organized by subdirectory (label).
    
    Args:
        directory: Root directory containing label subdirectories
        
    Returns:
        Dict mapping label names to lists of file paths
    """
    label_files = defaultdict(list)
    
    for item in directory.iterdir():
        if item.is_dir() and not item.name.startswith('.'):
            label = item.name
            # Find all readable files in this subdirectory
            for file in item.rglob('*'):
                if file.is_file() and file.suffix.lower() in ['.txt', '.text', '.md', '']:
                    label_files[label].append(file)
                elif file.is_file():
                    # Try to read other files as well
                    try:
                        with open(file, 'r', encoding='utf-8', errors='ignore') as f:
                            _ = f.read(100)  # Test read
                        label_files[label].append(file)
                    except:
                        pass
    
    return dict(label_files)


def read_file_content(file_path: Path, max_chars: int = 8000) -> Optional[str]:
    """Read file content, truncating if necessary."""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read(max_chars)
        return content.strip()
    except Exception as e:
        logger.warning(f"Could not read {file_path}: {e}")
        return None


def run_batch_classification(
    input_dir: str,
    output_file: str,
    model_path: str = "output/lora_adapters/final",
    base_model: str = "unsloth/gemma-3-270m",
    output_format: str = "json",
    use_categories: bool = True
):
    """
    Run batch classification on a directory of documents.
    
    Args:
        input_dir: Directory containing label subdirectories
        output_file: Output file path (JSON or CSV)
        model_path: Path to LoRA adapter directory
        base_model: HuggingFace base model name
        output_format: 'json' or 'csv'
        use_categories: Whether to provide category list to model
    """
    input_path = Path(input_dir)
    if not input_path.exists():
        logger.error(f"Input directory does not exist: {input_dir}")
        sys.exit(1)
    
    # Find all files organized by label
    logger.info(f"Scanning directory: {input_dir}")
    label_files = find_text_files(input_path)
    
    if not label_files:
        logger.error("No subdirectories with files found!")
        sys.exit(1)
    
    # Summary
    total_files = sum(len(files) for files in label_files.values())
    labels = list(label_files.keys())
    
    logger.info(f"Found {total_files} files across {len(labels)} categories")
    for label, files in label_files.items():
        logger.info(f"  - {label}: {len(files)} files")
    
    # Load classifier
    logger.info(f"Loading model from: {model_path}")
    from inference.predictor import DocumentClassifier
    
    classifier = DocumentClassifier(
        model_path=model_path,
        base_model_name=base_model
    )
    logger.info(f"Model loaded on {classifier.device}")
    
    # Run classification
    results = []
    correct = 0
    per_label_stats = defaultdict(lambda: {"total": 0, "correct": 0})
    
    categories = labels if use_categories else None
    
    logger.info("Starting classification...")
    
    for label, files in label_files.items():
        for file_path in files:
            content = read_file_content(file_path)
            if content is None:
                continue
            
            # Classify
            try:
                prediction = classifier.predict(
                    text=content,
                    categories=categories,
                    return_raw=True
                )
                predicted_label = prediction.get('label', 'UNKNOWN')
                is_correct = predicted_label == label
                
                result = {
                    "file": str(file_path),
                    "ground_truth": label,
                    "predicted": predicted_label,
                    "correct": is_correct,
                    "raw_output": prediction.get('raw_output', '')
                }
                results.append(result)
                
                # Stats
                if is_correct:
                    correct += 1
                per_label_stats[label]["total"] += 1
                if is_correct:
                    per_label_stats[label]["correct"] += 1
                
                status = "✓" if is_correct else "✗"
                logger.info(f"  {status} {file_path.name}: {label} -> {predicted_label}")
                
            except Exception as e:
                logger.error(f"Error classifying {file_path}: {e}")
                results.append({
                    "file": str(file_path),
                    "ground_truth": label,
                    "predicted": "ERROR",
                    "correct": False,
                    "error": str(e)
                })
    
    # Calculate metrics
    total_classified = len([r for r in results if r.get('predicted') != 'ERROR'])
    accuracy = correct / total_classified if total_classified > 0 else 0
    
    # Per-label accuracy
    per_label_accuracy = {}
    for label, stats in per_label_stats.items():
        if stats["total"] > 0:
            per_label_accuracy[label] = {
                "accuracy": stats["correct"] / stats["total"],
                "correct": stats["correct"],
                "total": stats["total"]
            }
    
    # Summary
    summary = {
        "timestamp": datetime.now().isoformat(),
        "input_directory": str(input_path.absolute()),
        "model_path": model_path,
        "total_files": total_classified,
        "correct_predictions": correct,
        "overall_accuracy": round(accuracy * 100, 2),
        "categories_used": labels,
        "per_category_accuracy": per_label_accuracy
    }
    
    # Output results
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if output_format == 'json':
        output_data = {
            "summary": summary,
            "results": results
        }
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
    else:  # CSV
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=[
                "file", "ground_truth", "predicted", "correct", "raw_output"
            ])
            writer.writeheader()
            for r in results:
                row = {k: r.get(k, '') for k in writer.fieldnames}
                writer.writerow(row)
        
        # Also write summary JSON
        summary_path = output_path.with_suffix('.summary.json')
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
    
    logger.info(f"\nResults saved to: {output_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("CLASSIFICATION SUMMARY")
    print("="*60)
    print(f"Total files processed: {total_classified}")
    print(f"Correct predictions:   {correct}")
    print(f"Overall accuracy:      {accuracy*100:.2f}%")
    print("\nPer-category breakdown:")
    for label, stats in sorted(per_label_accuracy.items()):
        acc = stats['accuracy'] * 100
        print(f"  {label:30s} {stats['correct']:3d}/{stats['total']:3d} ({acc:.1f}%)")
    print("="*60)
    
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Batch classify documents from a directory structure",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
    python scripts/batch_classify.py \\
        --input_dir ./test_data \\
        --output_file ./results.json \\
        --model_path ./output/lora_adapters/final

Directory structure:
    test_data/
    ├── OfferLetter/
    │   ├── doc1.txt
    │   └── doc2.txt
    └── SalaryStructure/
        └── salary_doc.txt
        """
    )
    
    parser.add_argument(
        "--input_dir", "-i",
        required=True,
        help="Directory containing label subdirectories with document files"
    )
    
    parser.add_argument(
        "--output_file", "-o",
        default="batch_results.json",
        help="Output file path (default: batch_results.json)"
    )
    
    parser.add_argument(
        "--model_path", "-m",
        default="output/lora_adapters/final",
        help="Path to LoRA adapter directory (default: output/lora_adapters/final)"
    )
    
    parser.add_argument(
        "--base_model", "-b",
        default="unsloth/gemma-3-270m",
        help="HuggingFace base model name (default: unsloth/gemma-3-270m)"
    )
    
    parser.add_argument(
        "--format", "-f",
        choices=["json", "csv"],
        default="json",
        help="Output format (default: json)"
    )
    
    parser.add_argument(
        "--no-categories",
        action="store_true",
        help="Don't provide category list to model (uses detected labels by default)"
    )
    
    args = parser.parse_args()
    
    run_batch_classification(
        input_dir=args.input_dir,
        output_file=args.output_file,
        model_path=args.model_path,
        base_model=args.base_model,
        output_format=args.format,
        use_categories=not args.no_categories
    )


if __name__ == "__main__":
    main()
