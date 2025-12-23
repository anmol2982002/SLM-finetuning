"""
Custom Dataset for Causal Language Model Fine-Tuning
=====================================================
Handles tokenization and proper label masking for instruction-tuned classification.
"""

import json
import logging
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class InstructionDataset(Dataset):
    """
    Dataset for instruction-style classification training.
    
    Features:
    - Tokenizes instruction + response
    - Masks instruction tokens (loss only on response)
    - Ensures response is NEVER truncated (critical for classification)
    - Handles padding and truncation
    """
    
    def __init__(
        self,
        data_file: str,
        tokenizer,
        max_length: int = 2048,
        mask_instruction: bool = True
    ):
        """
        Initialize dataset.
        
        Args:
            data_file: Path to JSONL file with instruction/response pairs
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length
            mask_instruction: If True, mask instruction tokens in labels
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mask_instruction = mask_instruction
        
        # Load data
        self.data = []
        with open(data_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    self.data.append(json.loads(line))
        
        logger.info(f"Loaded {len(self.data)} samples from {data_file}")
        
        # Ensure tokenizer has padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # Reserve tokens for response (response is short: {"label": "..."} + end_of_turn)
        # Response is typically ~20-30 tokens max
        self.response_reserve = 50
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        instruction = item['instruction']
        response = item['response']
        
        # Add end_of_turn to response
        response_with_end = response + "<end_of_turn>"
        
        # Tokenize response first (we need to know its length to reserve space)
        response_encoded = self.tokenizer(
            response_with_end,
            add_special_tokens=False,  # Don't add BOS here
            return_tensors='pt'
        )
        response_ids = response_encoded['input_ids'].squeeze(0)
        response_length = len(response_ids)
        
        # Calculate max length for instruction (reserve space for response)
        max_instruction_length = self.max_length - response_length - 1  # -1 for safety
        
        # Tokenize instruction with truncation
        instruction_encoded = self.tokenizer(
            instruction,
            max_length=max_instruction_length,
            truncation=True,
            add_special_tokens=True,  # Add BOS token
            return_tensors='pt'
        )
        instruction_ids = instruction_encoded['input_ids'].squeeze(0)
        instruction_length = len(instruction_ids)
        
        # Concatenate instruction + response
        input_ids = torch.cat([instruction_ids, response_ids])
        
        # Create attention mask (all 1s for actual tokens)
        attention_mask = torch.ones(len(input_ids), dtype=torch.long)
        
        # Pad to max_length
        current_length = len(input_ids)
        if current_length < self.max_length:
            pad_length = self.max_length - current_length
            input_ids = torch.cat([
                input_ids,
                torch.full((pad_length,), self.tokenizer.pad_token_id, dtype=torch.long)
            ])
            attention_mask = torch.cat([
                attention_mask,
                torch.zeros(pad_length, dtype=torch.long)
            ])
        elif current_length > self.max_length:
            # This shouldn't happen, but handle it
            input_ids = input_ids[:self.max_length]
            attention_mask = attention_mask[:self.max_length]
            # Ensure we still have response tokens
            instruction_length = min(instruction_length, self.max_length - response_length)
        
        # Create labels (same as input_ids for causal LM)
        labels = input_ids.clone()
        
        if self.mask_instruction:
            # Mask instruction tokens (set to -100 so they're ignored in loss)
            # Only mask up to instruction_length, leaving response tokens unmasked
            labels[:instruction_length] = -100
        
        # Also mask padding tokens
        labels[attention_mask == 0] = -100
        
        # Sanity check: ensure we have some unmasked labels
        num_valid_labels = (labels != -100).sum().item()
        if num_valid_labels == 0:
            logger.warning(f"Sample {idx} has no valid labels! "
                          f"Instruction length: {instruction_length}, "
                          f"Response length: {response_length}")
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }


class DataCollatorForInstructionTuning:
    """
    Data collator that handles dynamic padding for instruction tuning.
    More memory efficient than padding to max_length.
    """
    
    def __init__(self, tokenizer, padding: bool = True):
        self.tokenizer = tokenizer
        self.padding = padding
    
    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        # Find max length in batch (excluding padding)
        max_len = max(
            (f['attention_mask'] == 1).sum().item() 
            for f in features
        )
        
        batch = {
            'input_ids': [],
            'attention_mask': [],
            'labels': []
        }
        
        for f in features:
            # Get actual sequence length (non-padding)
            actual_len = (f['attention_mask'] == 1).sum().item()
            
            # Trim to actual length first, then pad to max_len
            input_ids = f['input_ids'][:actual_len]
            attention_mask = f['attention_mask'][:actual_len]
            labels = f['labels'][:actual_len]
            
            pad_len = max_len - actual_len
            
            if pad_len > 0 and self.padding:
                # Pad on the right
                input_ids = torch.cat([
                    input_ids,
                    torch.full((pad_len,), self.tokenizer.pad_token_id, dtype=torch.long)
                ])
                attention_mask = torch.cat([
                    attention_mask,
                    torch.zeros(pad_len, dtype=torch.long)
                ])
                labels = torch.cat([
                    labels,
                    torch.full((pad_len,), -100, dtype=torch.long)
                ])
            
            batch['input_ids'].append(input_ids)
            batch['attention_mask'].append(attention_mask)
            batch['labels'].append(labels)
        
        # Stack into tensors
        return {
            'input_ids': torch.stack(batch['input_ids']),
            'attention_mask': torch.stack(batch['attention_mask']),
            'labels': torch.stack(batch['labels'])
        }
