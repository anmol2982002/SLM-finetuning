#!/usr/bin/env python3
"""
Dataset Preparation Script
==========================
Converts extracted text into instruction-style training format with train/val/test splits.

Input format (from extract_text.py):
    {"label": "OfferLetter", "text": "...", "source": "..."}

Output format (instruction-style for causal LM):
    {"instruction": "...", "response": "{\"label\": \"OfferLetter\"}"}

Usage:
    python scripts/prepare_dataset.py --input extracted.jsonl --output_dir data/processed/
"""

import os
import sys
import json
import argparse
import logging
import random
from pathlib import Path
from collections import Counter
from typing import Dict, List, Tuple, Optional

from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# PROMPT TEMPLATES
# ============================================================================

SYSTEM_PROMPT = """You are a document classification assistant. Your task is to classify documents into exactly ONE subcategory.

RULES:
1. Read the document carefully
2. Choose exactly ONE subcategory from the valid list
3. Output ONLY valid JSON in this exact format: {"label": "<subcategory>"}
4. Do not include any explanations, comments, or additional text
5. The label must exactly match one of the valid subcategories"""

CLASSIFICATION_INSTRUCTION = """<bos><start_of_turn>user
{system_prompt}

Valid subcategories:
{categories}

---
DOCUMENT:
{document}
---

Classify this document into exactly one subcategory.<end_of_turn>
<start_of_turn>model
"""

# For inference (without categories list for brevity, relies on training)
INFERENCE_INSTRUCTION = """<bos><start_of_turn>user
You are a document classification assistant. Classify the following document into exactly ONE subcategory.
Output ONLY valid JSON: {{"label": "<subcategory>"}}

DOCUMENT:
{document}

Classify this document.<end_of_turn>
<start_of_turn>model
"""


def load_extracted_data(input_file: str) -> List[Dict]:
    """Load extracted data from JSONL file."""
    data = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                if 'label' in item and 'text' in item:
                    data.append(item)
                else:
                    logger.warning(f"Line {line_num}: Missing 'label' or 'text' field")
            except json.JSONDecodeError as e:
                logger.warning(f"Line {line_num}: Invalid JSON - {e}")
    
    logger.info(f"Loaded {len(data)} records from {input_file}")
    return data


def truncate_text(text: str, max_chars: int = 12000) -> str:
    """
    Truncate text to fit within token limits.
    Rough estimate: 1 token ≈ 4 characters for English text.
    For 4096 max tokens, ~12000 chars is safe with room for prompt.
    """
    if len(text) <= max_chars:
        return text
    
    # Truncate and add indicator
    return text[:max_chars] + "\n\n[Document truncated...]"


def format_instruction_sample(
    text: str, 
    label: str, 
    categories: List[str],
    max_text_chars: int = 12000,
    include_categories: bool = True
) -> Dict:
    """
    Format a single sample into instruction-style format.
    
    Args:
        text: Document text
        label: True label
        categories: List of all valid categories
        max_text_chars: Maximum characters for document text
        include_categories: Whether to include category list in prompt
        
    Returns:
        Dict with 'instruction' and 'response' keys
    """
    # Truncate long documents
    text = truncate_text(text, max_text_chars)
    
    # Format categories as comma-separated list (sorted for consistency)
    categories_str = ", ".join(sorted(categories))
    
    # Build instruction
    if include_categories:
        instruction = CLASSIFICATION_INSTRUCTION.format(
            system_prompt=SYSTEM_PROMPT,
            categories=categories_str,
            document=text
        )
    else:
        instruction = INFERENCE_INSTRUCTION.format(document=text)
    
    # Build response (JSON only)
    response = json.dumps({"label": label}, ensure_ascii=False)
    
    return {
        'instruction': instruction,
        'response': response
    }


def stratified_split(
    data: List[Dict],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Split data into train/val/test with stratification by label.
    
    Args:
        data: List of data items with 'label' field
        train_ratio: Proportion for training
        val_ratio: Proportion for validation
        test_ratio: Proportion for testing
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_data, val_data, test_data)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"
    
    random.seed(seed)
    
    # Group by label
    by_label = {}
    for item in data:
        label = item['label']
        if label not in by_label:
            by_label[label] = []
        by_label[label].append(item)
    
    train_data = []
    val_data = []
    test_data = []
    
    for label, items in by_label.items():
        # Shuffle within label
        random.shuffle(items)
        
        n = len(items)
        n_train = max(1, int(n * train_ratio))
        n_val = max(1, int(n * val_ratio))
        n_test = n - n_train - n_val
        
        # Ensure at least 1 sample in each split if possible
        if n_test < 1 and n > 2:
            n_test = 1
            n_train = n - n_val - n_test
        
        train_data.extend(items[:n_train])
        val_data.extend(items[n_train:n_train + n_val])
        test_data.extend(items[n_train + n_val:])
    
    # Shuffle each split
    random.shuffle(train_data)
    random.shuffle(val_data)
    random.shuffle(test_data)
    
    return train_data, val_data, test_data


def prepare_dataset(
    input_file: str,
    output_dir: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    max_text_chars: int = 12000,
    seed: int = 42,
    validate_only: bool = False
) -> Dict:
    """
    Prepare instruction-style dataset with train/val/test splits.
    
    Args:
        input_file: Path to extracted JSONL file
        output_dir: Directory to save processed datasets
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
        max_text_chars: Maximum document text length
        seed: Random seed
        validate_only: Only validate, don't write files
        
    Returns:
        Statistics dictionary
    """
    # Load data
    data = load_extracted_data(input_file)
    
    if not data:
        logger.error("No valid data loaded")
        sys.exit(1)
    
    # Get all unique labels
    labels = sorted(set(item['label'] for item in data))
    label_counts = Counter(item['label'] for item in data)
    
    logger.info(f"Found {len(labels)} unique labels")
    
    # Split data
    train_data, val_data, test_data = stratified_split(
        data, train_ratio, val_ratio, test_ratio, seed
    )
    
    stats = {
        'total_samples': len(data),
        'num_labels': len(labels),
        'labels': labels,
        'label_counts': dict(label_counts),
        'train_samples': len(train_data),
        'val_samples': len(val_data),
        'test_samples': len(test_data),
        'max_text_chars': max_text_chars
    }
    
    if validate_only:
        print("\n=== Dataset Validation ===")
        print(f"Total samples: {stats['total_samples']}")
        print(f"Number of labels: {stats['num_labels']}")
        print(f"Train/Val/Test: {stats['train_samples']}/{stats['val_samples']}/{stats['test_samples']}")
        print("\nLabel distribution:")
        for label, count in sorted(label_counts.items()):
            print(f"  {label}: {count}")
        return stats
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Process and write each split
    splits = [
        ('train', train_data),
        ('val', val_data),
        ('test', test_data)
    ]
    
    for split_name, split_data in splits:
        output_file = output_path / f'{split_name}.jsonl'
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in tqdm(split_data, desc=f"Processing {split_name}"):
                formatted = format_instruction_sample(
                    text=item['text'],
                    label=item['label'],
                    categories=labels,
                    max_text_chars=max_text_chars
                )
                f.write(json.dumps(formatted, ensure_ascii=False) + '\n')
        
        logger.info(f"Wrote {len(split_data)} samples to {output_file}")
    
    # Save label mapping
    label_mapping = {label: idx for idx, label in enumerate(labels)}
    mapping_file = output_path / 'label_mapping.json'
    with open(mapping_file, 'w', encoding='utf-8') as f:
        json.dump({
            'label_to_id': label_mapping,
            'id_to_label': {v: k for k, v in label_mapping.items()},
            'labels': labels
        }, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Saved label mapping to {mapping_file}")
    
    # Save statistics
    stats_file = output_path / 'dataset_stats.json'
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    # Print summary
    print("\n" + "=" * 50)
    print("DATASET PREPARATION SUMMARY")
    print("=" * 50)
    print(f"Total samples: {stats['total_samples']}")
    print(f"Number of labels: {stats['num_labels']}")
    print(f"\nSplit sizes:")
    print(f"  Train: {stats['train_samples']} ({train_ratio*100:.0f}%)")
    print(f"  Val:   {stats['val_samples']} ({val_ratio*100:.0f}%)")
    print(f"  Test:  {stats['test_samples']} ({test_ratio*100:.0f}%)")
    print(f"\nOutput files:")
    print(f"  {output_path / 'train.jsonl'}")
    print(f"  {output_path / 'val.jsonl'}")
    print(f"  {output_path / 'test.jsonl'}")
    print(f"  {output_path / 'label_mapping.json'}")
    print("=" * 50)
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description='Prepare instruction-style dataset for classification',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python scripts/prepare_dataset.py --input extracted.jsonl --output_dir data/processed/
  
  # Custom split ratios
  python scripts/prepare_dataset.py --input extracted.jsonl --output_dir data/processed/ \\
      --train-ratio 0.7 --val-ratio 0.15 --test-ratio 0.15
  
  # Validate only
  python scripts/prepare_dataset.py --input extracted.jsonl --output_dir data/processed/ --validate
        """
    )
    
    parser.add_argument(
        '--input', '-i',
        required=True,
        help='Input JSONL file from extract_text.py'
    )
    parser.add_argument(
        '--output_dir', '-o',
        required=True,
        help='Output directory for processed datasets'
    )
    parser.add_argument(
        '--train-ratio',
        type=float,
        default=0.8,
        help='Training set ratio (default: 0.8)'
    )
    parser.add_argument(
        '--val-ratio',
        type=float,
        default=0.1,
        help='Validation set ratio (default: 0.1)'
    )
    parser.add_argument(
        '--test-ratio',
        type=float,
        default=0.1,
        help='Test set ratio (default: 0.1)'
    )
    parser.add_argument(
        '--max-text-chars',
        type=int,
        default=12000,
        help='Maximum document text length in characters (default: 12000)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    parser.add_argument(
        '--validate',
        action='store_true',
        help='Only validate, do not write files'
    )
    
    args = parser.parse_args()
    
    # Validate ratios
    if abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) > 1e-6:
        logger.error("Split ratios must sum to 1.0")
        sys.exit(1)
    
    prepare_dataset(
        input_file=args.input,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        max_text_chars=args.max_text_chars,
        seed=args.seed,
        validate_only=args.validate
    )


if __name__ == '__main__':
    main()
