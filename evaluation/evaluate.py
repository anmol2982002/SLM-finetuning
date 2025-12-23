#!/usr/bin/env python3
"""
Model Evaluation Script
=======================
Evaluate fine-tuned model on test set with comprehensive metrics.

Usage:
    python evaluation/evaluate.py --model_path output/lora_adapters/final --test_file data/processed/test.jsonl
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from evaluation.metrics import (
    parse_json_prediction,
    compute_metrics,
    compute_json_validity_metrics,
    format_metrics_report,
    save_confusion_matrix
)
from training.prompts import format_inference_prompt

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Evaluator:
    """Handles model loading and evaluation."""
    
    def __init__(
        self,
        model_path: str,
        base_model_name: str = "unsloth/gemma-3-270m",
        device: Optional[str] = None,
        categories: Optional[List[str]] = None
    ):
        """
        Initialize evaluator with model.
        
        Args:
            model_path: Path to LoRA adapter directory
            base_model_name: HuggingFace model name for base model
            device: Device to use ('cuda', 'cpu', or None for auto)
            categories: List of valid category names (REQUIRED for correct predictions)
        """
        self.model_path = model_path
        self.base_model_name = base_model_name
        self.categories = categories or []
        
        # Setup device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        logger.info(f"Using device: {self.device}")
        
        # Load model and tokenizer
        self._load_model()
    
    def _load_model(self):
        """Load tokenizer and model with LoRA adapters."""
        logger.info(f"Loading tokenizer from {self.model_path}")
        
        # Try loading tokenizer from adapter path first
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                use_fast=True,
                trust_remote_code=True
            )
        except:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.base_model_name,
                use_fast=True,
                trust_remote_code=True
            )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        logger.info(f"Loading base model from {self.base_model_name}")
        
        # Load base model with bf16 (must match training dtype)
        if self.device == 'cuda' and torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
        elif self.device == 'cuda':
            dtype = torch.float16
        else:
            dtype = torch.float32
            
        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            torch_dtype=dtype,
            device_map="auto" if self.device == 'cuda' else None,
            trust_remote_code=True
        )
        
        # Resize embeddings to match tokenizer (training may have added special tokens)
        if len(self.tokenizer) != base_model.get_input_embeddings().weight.shape[0]:
            logger.info(f"Resizing embeddings: {base_model.get_input_embeddings().weight.shape[0]} -> {len(self.tokenizer)}")
            base_model.resize_token_embeddings(len(self.tokenizer))
        
        # Load LoRA adapters
        logger.info(f"Loading LoRA adapters from {self.model_path}")
        self.model = PeftModel.from_pretrained(base_model, self.model_path)
        
        if self.device == 'cpu':
            self.model = self.model.to(self.device)
        
        self.model.eval()
        logger.info("Model loaded successfully")
    
    def predict(
        self,
        text: str,
        max_new_tokens: int = 50,
        temperature: float = 0.0
    ) -> str:
        """
        Generate prediction for a single document.
        
        Args:
            text: Document text
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0 = greedy)
            
        Returns:
            Generated text
        """
        # Format prompt with categories (critical for correct predictions!)
        prompt = format_inference_prompt(text, categories=self.categories)
        
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors='pt',
            truncation=True,
            max_length=4096
        ).to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else None,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode only new tokens
        generated = outputs[0][inputs['input_ids'].shape[1]:]
        result = self.tokenizer.decode(generated, skip_special_tokens=True)
        
        return result.strip()
    
    def predict_batch(
        self,
        texts: List[str],
        max_new_tokens: int = 50,
        show_progress: bool = True
    ) -> List[str]:
        """
        Generate predictions for multiple documents.
        
        Args:
            texts: List of document texts
            max_new_tokens: Maximum tokens to generate
            show_progress: Show progress bar
            
        Returns:
            List of generated texts
        """
        predictions = []
        iterator = tqdm(texts, desc="Predicting") if show_progress else texts
        
        for text in iterator:
            pred = self.predict(text, max_new_tokens)
            predictions.append(pred)
        
        return predictions


def load_test_data(test_file: str) -> Tuple[List[str], List[str]]:
    """
    Load test data from JSONL file.
    
    Args:
        test_file: Path to test JSONL file
        
    Returns:
        Tuple of (documents, labels)
    """
    documents = []
    labels = []
    
    with open(test_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            item = json.loads(line)
            instruction = item.get('instruction', '')
            response = item.get('response', '')
            
            # Extract document from instruction
            # The document is between "DOCUMENT:" and the next delimiter
            import re
            doc_match = re.search(r'DOCUMENT:\n(.+?)(?:\n---|\n\nClassify)', instruction, re.DOTALL)
            if doc_match:
                documents.append(doc_match.group(1).strip())
            else:
                # Fallback: use full instruction
                documents.append(instruction)
            
            # Extract label from response
            try:
                resp_data = json.loads(response)
                labels.append(resp_data.get('label', 'UNKNOWN'))
            except:
                labels.append('UNKNOWN')
    
    logger.info(f"Loaded {len(documents)} test samples")
    return documents, labels


def run_evaluation(
    model_path: str,
    test_file: str,
    output_dir: str,
    base_model_name: str = "unsloth/gemma-3-270m",
    max_samples: Optional[int] = None
):
    """
    Run full evaluation pipeline.
    
    Args:
        model_path: Path to LoRA adapter directory
        test_file: Path to test JSONL file
        output_dir: Directory to save results
        base_model_name: Base model name
        max_samples: Optional limit on samples to evaluate
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load test data FIRST to get categories
    documents, ground_truth = load_test_data(test_file)
    
    if max_samples:
        documents = documents[:max_samples]
        ground_truth = ground_truth[:max_samples]
        logger.info(f"Limited to {max_samples} samples")
    
    # Get label mapping if available (BEFORE creating evaluator)
    label_mapping_file = Path(test_file).parent / 'label_mapping.json'
    if label_mapping_file.exists():
        with open(label_mapping_file, 'r') as f:
            mapping = json.load(f)
            all_labels = mapping.get('labels', sorted(set(ground_truth)))
    else:
        all_labels = sorted(set(ground_truth))
    
    logger.info(f"Evaluating on {len(documents)} samples with {len(all_labels)} labels")
    logger.info(f"Categories: {all_labels}")
    
    # Load evaluator WITH categories (critical for correct predictions!)
    evaluator = Evaluator(model_path, base_model_name, categories=all_labels)
    
    # Generate predictions
    logger.info("Generating predictions...")
    raw_predictions = evaluator.predict_batch(documents)
    
    # Parse predictions to extract labels
    parsed_predictions = []
    for raw in raw_predictions:
        label, _ = parse_json_prediction(raw)
        parsed_predictions.append(label)
    
    # Compute metrics
    logger.info("Computing metrics...")
    metrics = compute_metrics(parsed_predictions, ground_truth, all_labels)
    json_metrics = compute_json_validity_metrics(raw_predictions)
    
    # Format and print report
    report = format_metrics_report(metrics, json_metrics)
    print("\n" + report)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save metrics JSON
    metrics_file = output_path / f'metrics_{timestamp}.json'
    with open(metrics_file, 'w', encoding='utf-8') as f:
        # Remove non-serializable items
        save_metrics = {k: v for k, v in metrics.items() if k != 'classification_report'}
        save_metrics['json_validity'] = json_metrics
        json.dump(save_metrics, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved metrics to {metrics_file}")
    
    # Save text report
    report_file = output_path / f'report_{timestamp}.txt'
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    logger.info(f"Saved report to {report_file}")
    
    # Save confusion matrix
    cm_file = output_path / f'confusion_matrix_{timestamp}.csv'
    save_confusion_matrix(metrics['confusion_matrix'], metrics['labels'], str(cm_file))
    
    # Save detailed predictions
    predictions_file = output_path / f'predictions_{timestamp}.jsonl'
    with open(predictions_file, 'w', encoding='utf-8') as f:
        for i, (doc, truth, raw, parsed) in enumerate(zip(
            documents, ground_truth, raw_predictions, parsed_predictions
        )):
            item = {
                'index': i,
                'document_snippet': doc[:500],
                'ground_truth': truth,
                'raw_prediction': raw,
                'parsed_label': parsed,
                'correct': truth == parsed
            }
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    logger.info(f"Saved predictions to {predictions_file}")
    
    # Save misclassified samples
    misclassified = []
    for i, (doc, truth, parsed) in enumerate(zip(documents, ground_truth, parsed_predictions)):
        if truth != parsed:
            misclassified.append({
                'index': i,
                'document_snippet': doc[:500],
                'ground_truth': truth,
                'predicted': parsed
            })
    
    errors_file = output_path / f'errors_{timestamp}.jsonl'
    with open(errors_file, 'w', encoding='utf-8') as f:
        for item in misclassified:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    logger.info(f"Saved {len(misclassified)} errors to {errors_file}")
    
    print(f"\n✅ Evaluation complete! Results saved to: {output_path}")
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate fine-tuned document classification model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic evaluation
  python evaluation/evaluate.py --model_path output/lora_adapters/final --test_file data/processed/test.jsonl
  
  # With custom output directory
  python evaluation/evaluate.py --model_path output/lora_adapters/final --test_file data/processed/test.jsonl --output_dir eval_results/
  
  # Quick test with limited samples
  python evaluation/evaluate.py --model_path output/lora_adapters/final --test_file data/processed/test.jsonl --max-samples 100
        """
    )
    
    parser.add_argument(
        '--model_path', '-m',
        required=True,
        help='Path to LoRA adapter directory'
    )
    parser.add_argument(
        '--test_file', '-t',
        required=True,
        help='Path to test JSONL file'
    )
    parser.add_argument(
        '--output_dir', '-o',
        default='eval_results',
        help='Directory to save evaluation results (default: eval_results)'
    )
    parser.add_argument(
        '--base_model',
        default='unsloth/gemma-3-270m',
        help='Base model name (default: unsloth/gemma-3-270m)'
    )
    parser.add_argument(
        '--max-samples',
        type=int,
        default=None,
        help='Maximum samples to evaluate (for quick testing)'
    )
    
    args = parser.parse_args()
    
    run_evaluation(
        model_path=args.model_path,
        test_file=args.test_file,
        output_dir=args.output_dir,
        base_model_name=args.base_model,
        max_samples=args.max_samples
    )


if __name__ == '__main__':
    main()
