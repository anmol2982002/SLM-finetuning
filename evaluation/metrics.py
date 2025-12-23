"""
Metrics Computation for Document Classification
================================================
Comprehensive metrics including per-class scores, confusion matrix,
and JSON validity checks.
"""

import json
import logging
from typing import Dict, List, Tuple, Optional
from collections import Counter

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)

logger = logging.getLogger(__name__)


def parse_json_prediction(text: str) -> Tuple[Optional[str], bool]:
    """
    Extract label from model prediction text.
    
    Args:
        text: Model output text
        
    Returns:
        Tuple of (extracted_label, is_valid_json)
    """
    import re
    
    # Try to find JSON in the text
    # Pattern: {"label": "..."}
    pattern = r'\{\s*"label"\s*:\s*"([^"]+)"\s*\}'
    match = re.search(pattern, text)
    
    if match:
        return match.group(1), True
    
    # Try parsing as JSON directly
    try:
        data = json.loads(text.strip())
        if isinstance(data, dict) and 'label' in data:
            return data['label'], True
    except json.JSONDecodeError:
        pass
    
    # Fallback: look for any label-like pattern
    # Sometimes model outputs: label: SomeCategory
    pattern2 = r'label["\s:]+([A-Za-z][A-Za-z0-9_]+)'
    match2 = re.search(pattern2, text, re.IGNORECASE)
    if match2:
        return match2.group(1), False  # Valid label but not JSON format
    
    return None, False


def compute_metrics(
    predictions: List[str],
    ground_truth: List[str],
    labels: Optional[List[str]] = None
) -> Dict:
    """
    Compute comprehensive classification metrics.
    
    Args:
        predictions: List of predicted labels
        ground_truth: List of true labels
        labels: Optional list of all possible labels (for ordering)
        
    Returns:
        Dictionary with all metrics
    """
    # Get all unique labels if not provided
    if labels is None:
        labels = sorted(set(ground_truth) | set(predictions))
    
    # Filter out predictions that don't match any known label
    valid_preds = []
    valid_truth = []
    unknown_count = 0
    
    for pred, truth in zip(predictions, ground_truth):
        if pred is None or pred not in labels:
            unknown_count += 1
            # Use a placeholder for unknown predictions
            pred = "__UNKNOWN__"
        valid_preds.append(pred)
        valid_truth.append(truth)
    
    # Add unknown to labels if needed
    if unknown_count > 0:
        labels = labels + ["__UNKNOWN__"]
    
    # Accuracy
    accuracy = accuracy_score(valid_truth, valid_preds)
    
    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        valid_truth, valid_preds, labels=labels, zero_division=0
    )
    
    # Macro and weighted averages
    macro_precision = np.mean(precision[:-1] if unknown_count > 0 else precision)
    macro_recall = np.mean(recall[:-1] if unknown_count > 0 else recall)
    macro_f1 = np.mean(f1[:-1] if unknown_count > 0 else f1)
    
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        valid_truth, valid_preds, average='weighted', zero_division=0
    )
    
    # Per-class breakdown
    per_class = {}
    for i, label in enumerate(labels):
        if label == "__UNKNOWN__":
            continue
        per_class[label] = {
            'precision': float(precision[i]),
            'recall': float(recall[i]),
            'f1': float(f1[i]),
            'support': int(support[i])
        }
    
    # Classification report (as string)
    report = classification_report(
        valid_truth, valid_preds, labels=labels, zero_division=0
    )
    
    # Confusion matrix
    cm = confusion_matrix(valid_truth, valid_preds, labels=labels)
    
    return {
        'accuracy': float(accuracy),
        'macro_precision': float(macro_precision),
        'macro_recall': float(macro_recall),
        'macro_f1': float(macro_f1),
        'weighted_precision': float(weighted_precision),
        'weighted_recall': float(weighted_recall),
        'weighted_f1': float(weighted_f1),
        'per_class': per_class,
        'classification_report': report,
        'confusion_matrix': cm.tolist(),
        'labels': labels,
        'unknown_predictions': unknown_count,
        'total_samples': len(predictions)
    }


def compute_json_validity_metrics(
    raw_predictions: List[str]
) -> Dict:
    """
    Check JSON validity of model outputs.
    
    Args:
        raw_predictions: List of raw model output strings
        
    Returns:
        Dictionary with validity metrics
    """
    valid_json = 0
    valid_label = 0
    invalid_samples = []
    
    for i, pred in enumerate(raw_predictions):
        label, is_json = parse_json_prediction(pred)
        
        if is_json:
            valid_json += 1
        if label is not None:
            valid_label += 1
        else:
            invalid_samples.append({
                'index': i,
                'prediction': pred[:200]  # Truncate for readability
            })
    
    total = len(raw_predictions)
    
    return {
        'total_predictions': total,
        'valid_json': valid_json,
        'valid_json_rate': valid_json / total if total > 0 else 0,
        'valid_label': valid_label,
        'valid_label_rate': valid_label / total if total > 0 else 0,
        'invalid_samples': invalid_samples[:10]  # First 10 for debugging
    }


def format_metrics_report(metrics: Dict, json_metrics: Optional[Dict] = None) -> str:
    """
    Format metrics into a readable report.
    
    Args:
        metrics: Metrics from compute_metrics
        json_metrics: Optional metrics from compute_json_validity_metrics
        
    Returns:
        Formatted string report
    """
    lines = []
    lines.append("=" * 60)
    lines.append("EVALUATION RESULTS")
    lines.append("=" * 60)
    
    # Overall metrics
    lines.append("\n📊 OVERALL METRICS")
    lines.append("-" * 40)
    lines.append(f"Accuracy:           {metrics['accuracy']:.4f}")
    lines.append(f"Macro F1:           {metrics['macro_f1']:.4f}")
    lines.append(f"Macro Precision:    {metrics['macro_precision']:.4f}")
    lines.append(f"Macro Recall:       {metrics['macro_recall']:.4f}")
    lines.append(f"Weighted F1:        {metrics['weighted_f1']:.4f}")
    
    # JSON validity
    if json_metrics:
        lines.append("\n📝 JSON VALIDITY")
        lines.append("-" * 40)
        lines.append(f"Valid JSON:         {json_metrics['valid_json']}/{json_metrics['total_predictions']} ({json_metrics['valid_json_rate']*100:.1f}%)")
        lines.append(f"Valid Label:        {json_metrics['valid_label']}/{json_metrics['total_predictions']} ({json_metrics['valid_label_rate']*100:.1f}%)")
    
    # Unknown predictions
    if metrics['unknown_predictions'] > 0:
        lines.append(f"\n⚠️  Unknown predictions: {metrics['unknown_predictions']}")
    
    # Per-class metrics (top/bottom 5)
    lines.append("\n📈 PER-CLASS METRICS (sorted by F1)")
    lines.append("-" * 40)
    
    sorted_classes = sorted(
        metrics['per_class'].items(),
        key=lambda x: x[1]['f1'],
        reverse=True
    )
    
    # Show best 5
    lines.append("\nTop 5 performing classes:")
    for label, scores in sorted_classes[:5]:
        lines.append(f"  {label}: F1={scores['f1']:.3f}, P={scores['precision']:.3f}, R={scores['recall']:.3f} (n={scores['support']})")
    
    # Show worst 5
    if len(sorted_classes) > 10:
        lines.append("\nBottom 5 performing classes:")
        for label, scores in sorted_classes[-5:]:
            lines.append(f"  {label}: F1={scores['f1']:.3f}, P={scores['precision']:.3f}, R={scores['recall']:.3f} (n={scores['support']})")
    
    # Full classification report
    lines.append("\n📋 DETAILED CLASSIFICATION REPORT")
    lines.append("-" * 40)
    lines.append(metrics['classification_report'])
    
    lines.append("=" * 60)
    
    return "\n".join(lines)


def save_confusion_matrix(
    confusion_matrix: List[List[int]],
    labels: List[str],
    output_path: str
):
    """
    Save confusion matrix as a CSV file.
    
    Args:
        confusion_matrix: 2D list of confusion matrix values
        labels: List of label names
        output_path: Path to save CSV
    """
    import csv
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # Header row
        writer.writerow([''] + labels)
        
        # Data rows
        for i, row in enumerate(confusion_matrix):
            writer.writerow([labels[i]] + row)
    
    logger.info(f"Saved confusion matrix to {output_path}")
