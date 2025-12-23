#!/usr/bin/env python3
"""
LoRA Fine-Tuning Script for Document Classification
====================================================
Fine-tunes Gemma-3 270M with LoRA adapters for instruction-based classification.

Features:
- LoRA (Low-Rank Adaptation) with frozen base weights
- GPU/CPU auto-detection
- Gradient checkpointing for memory efficiency
- Wandb/TensorBoard logging support
- Checkpoint saving and resumption

Usage:
    python training/train_lora.py --config training/config.yaml
    
    # With custom output directory
    python training/train_lora.py --config training/config.yaml --output_dir output/my_experiment
    
    # Dry run (validate config)
    python training/train_lora.py --config training/config.yaml --dry-run
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Optional

import yaml
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
    set_seed
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from training.dataset import InstructionDataset, DataCollatorForInstructionTuning

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_device():
    """Detect and setup compute device."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        device = torch.device('cpu')
        logger.info("Using CPU - training will be slower")
        logger.warning("Consider using a GPU for faster training")
    
    return device


def load_tokenizer(model_name: str):
    """Load and configure tokenizer."""
    logger.info(f"Loading tokenizer from {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True,
        trust_remote_code=True
    )
    
    # Ensure padding token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        logger.info("Set pad_token to eos_token")
    
    return tokenizer


def load_model(model_name: str, config: dict, device: torch.device):
    """Load base model with optional quantization."""
    logger.info(f"Loading model from {model_name}")
    
    # Check if we should use quantization (for memory efficiency on GPU)
    use_quantization = config.get('use_quantization', False) and torch.cuda.is_available()
    
    if use_quantization:
        logger.info("Loading with 4-bit quantization")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
        model = prepare_model_for_kbit_training(model)
    else:
        # Load in bfloat16 for numerical stability (prevents nan loss on A100)
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
            logger.info("Using bfloat16 for numerical stability")
        elif torch.cuda.is_available():
            dtype = torch.float16
            logger.info("Using float16 (bf16 not supported)")
        else:
            dtype = torch.float32
            logger.info("Using float32 (CPU mode)")
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True
        )
        
        if device.type == 'cpu':
            model = model.to(device)
    
    # Enable gradient checkpointing for memory efficiency
    if config.get('gradient_checkpointing', True):
        model.gradient_checkpointing_enable()
        logger.info("Enabled gradient checkpointing")
    
    return model


def setup_lora(model, config: dict):
    """Configure and apply LoRA adapters."""
    lora_config = config.get('lora', {})
    
    peft_config = LoraConfig(
        r=lora_config.get('r', 16),
        lora_alpha=lora_config.get('alpha', 32),
        lora_dropout=lora_config.get('dropout', 0.05),
        target_modules=lora_config.get('target_modules', ['q_proj', 'k_proj', 'v_proj', 'o_proj']),
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    
    logger.info(f"LoRA config: r={peft_config.r}, alpha={peft_config.lora_alpha}, "
                f"dropout={peft_config.lora_dropout}")
    logger.info(f"Target modules: {peft_config.target_modules}")
    
    # Apply LoRA
    model = get_peft_model(model, peft_config)
    
    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable parameters: {trainable_params:,} / {total_params:,} "
                f"({100 * trainable_params / total_params:.2f}%)")
    
    return model


def setup_training_args(config: dict, output_dir: str, device: torch.device) -> TrainingArguments:
    """Configure training arguments."""
    
    # Determine if we can use fp16
    use_fp16 = torch.cuda.is_available()
    
    args = TrainingArguments(
        output_dir=output_dir,
        
        # Batch size (optimized for document classification)
        per_device_train_batch_size=config.get('per_device_train_batch_size', 2),
        per_device_eval_batch_size=config.get('per_device_eval_batch_size', 2),
        gradient_accumulation_steps=config.get('gradient_accumulation_steps', 8),
        
        # Training epochs & steps
        num_train_epochs=config.get('num_train_epochs', 3),
        max_steps=config.get('max_steps', -1),
        
        # Learning rate (optimized for LoRA fine-tuning)
        # Explicit float() to handle YAML parsing edge cases
        learning_rate=float(config.get('learning_rate', 2e-4)),
        lr_scheduler_type=config.get('lr_scheduler_type', 'cosine'),
        warmup_ratio=float(config.get('warmup_ratio', 0.1)),
        weight_decay=float(config.get('weight_decay', 0.01)),
        
        # Mixed precision
        # Use bf16 on A100/newer GPUs for numerical stability (fp16 can cause nan)
        fp16=False,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        
        # Logging
        logging_steps=config.get('logging_steps', 10),
        logging_first_step=True,
        report_to=config.get('report_to', 'none'),  # 'wandb', 'tensorboard', or 'none'
        
        # Evaluation
        eval_strategy="steps" if config.get('val_file') else "no",
        eval_steps=config.get('eval_steps', 100),
        
        # Saving
        save_strategy="steps",
        save_steps=config.get('save_steps', 100),
        save_total_limit=config.get('save_total_limit', 3),
        load_best_model_at_end=True if config.get('val_file') else False,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        
        # Memory optimization
        gradient_checkpointing=config.get('gradient_checkpointing', True),
        optim=config.get('optimizer', 'adamw_torch'),
        
        # Other
        seed=config.get('seed', 42),
        dataloader_num_workers=config.get('dataloader_num_workers', 0),
        remove_unused_columns=False,
        
        # Disable tqdm nesting issues
        disable_tqdm=False,
    )
    
    return args


def train(
    config_path: str,
    output_dir: Optional[str] = None,
    dry_run: bool = False,
    max_steps: Optional[int] = None
):
    """
    Main training function.
    
    Args:
        config_path: Path to configuration YAML
        output_dir: Override output directory
        dry_run: If True, only validate setup without training
        max_steps: Override max_steps for testing
    """
    # Load configuration
    config = load_config(config_path)
    
    # Set seed for reproducibility
    seed = config.get('seed', 42)
    set_seed(seed)
    logger.info(f"Set random seed: {seed}")
    
    # Setup device
    device = setup_device()
    
    # Determine output directory
    output_dir = output_dir or config.get('output_dir', 'output/lora_adapters')
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")
    
    # Load tokenizer
    model_name = config['model_name']
    tokenizer = load_tokenizer(model_name)
    
    # Load model
    model = load_model(model_name, config, device)
    
    # Resize embeddings if needed
    if len(tokenizer) > model.get_input_embeddings().weight.shape[0]:
        model.resize_token_embeddings(len(tokenizer))
    
    # Apply LoRA
    model = setup_lora(model, config)
    
    # Load datasets
    train_file = config['train_file']
    val_file = config.get('val_file')
    max_seq_length = config.get('max_seq_length', 4096)
    
    logger.info(f"Loading training data from {train_file}")
    train_dataset = InstructionDataset(
        data_file=train_file,
        tokenizer=tokenizer,
        max_length=max_seq_length,
        mask_instruction=True
    )
    
    eval_dataset = None
    if val_file and os.path.exists(val_file):
        logger.info(f"Loading validation data from {val_file}")
        eval_dataset = InstructionDataset(
            data_file=val_file,
            tokenizer=tokenizer,
            max_length=max_seq_length,
            mask_instruction=True
        )
    
    # Data collator
    data_collator = DataCollatorForInstructionTuning(tokenizer)
    
    # Override max_steps if provided
    if max_steps is not None:
        config['max_steps'] = max_steps
    
    # Training arguments
    training_args = setup_training_args(config, output_dir, device)
    
    if dry_run:
        logger.info("=" * 50)
        logger.info("DRY RUN - Configuration validated successfully")
        logger.info("=" * 50)
        logger.info(f"Model: {model_name}")
        logger.info(f"Training samples: {len(train_dataset)}")
        if eval_dataset:
            logger.info(f"Validation samples: {len(eval_dataset)}")
        logger.info(f"Max sequence length: {max_seq_length}")
        logger.info(f"Batch size per device: {training_args.per_device_train_batch_size}")
        logger.info(f"Gradient accumulation: {training_args.gradient_accumulation_steps}")
        logger.info(f"Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
        logger.info(f"Learning rate: {training_args.learning_rate}")
        logger.info(f"Epochs: {training_args.num_train_epochs}")
        return
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    # Train
    logger.info("=" * 50)
    logger.info("Starting training...")
    logger.info("=" * 50)
    
    try:
        trainer.train()
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user")
    
    # Save final model
    final_path = os.path.join(output_dir, 'final')
    logger.info(f"Saving final model to {final_path}")
    trainer.save_model(final_path)
    tokenizer.save_pretrained(final_path)
    
    # Save training config for reference
    config_save_path = os.path.join(final_path, 'training_config.yaml')
    with open(config_save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    logger.info("=" * 50)
    logger.info("Training complete!")
    logger.info(f"Model saved to: {final_path}")
    logger.info("=" * 50)


def main():
    parser = argparse.ArgumentParser(
        description='Fine-tune Gemma-3 with LoRA for document classification',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Standard training
  python training/train_lora.py --config training/config.yaml
  
  # With custom output directory
  python training/train_lora.py --config training/config.yaml --output_dir output/experiment1
  
  # Dry run to validate configuration
  python training/train_lora.py --config training/config.yaml --dry-run
  
  # Quick test run (5 steps)
  python training/train_lora.py --config training/config.yaml --max-steps 5
        """
    )
    
    parser.add_argument(
        '--config', '-c',
        required=True,
        help='Path to configuration YAML file'
    )
    parser.add_argument(
        '--output_dir', '-o',
        default=None,
        help='Override output directory from config'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Validate configuration without training'
    )
    parser.add_argument(
        '--max-steps',
        type=int,
        default=None,
        help='Override max training steps (for testing)'
    )
    
    args = parser.parse_args()
    
    train(
        config_path=args.config,
        output_dir=args.output_dir,
        dry_run=args.dry_run,
        max_steps=args.max_steps
    )


if __name__ == '__main__':
    main()