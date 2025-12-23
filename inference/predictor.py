"""
Document Classification Predictor
===================================
Core prediction logic for document classification using fine-tuned Gemma-3.
"""

import json
import re
import logging
from typing import Dict, List, Optional, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

logger = logging.getLogger(__name__)


class DocumentClassifier:
    """
    Document classifier using fine-tuned Gemma-3 with LoRA adapters.
    
    Features:
    - GPU/CPU auto-detection
    - Single and batch prediction
    - Few-shot inference for new categories
    - Confidence estimation
    """
    
    def __init__(
        self,
        model_path: str,
        base_model_name: str = "unsloth/gemma-3-270m",
        device: Optional[str] = None,
        load_in_8bit: bool = False
    ):
        """
        Initialize classifier.
        
        Args:
            model_path: Path to LoRA adapter directory
            base_model_name: HuggingFace model name for base model
            device: Device to use ('cuda', 'cpu', or None for auto)
            load_in_8bit: Load model in 8-bit quantization (requires bitsandbytes)
        """
        self.model_path = model_path
        self.base_model_name = base_model_name
        
        # Auto-detect device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        self.load_in_8bit = load_in_8bit and self.device == 'cuda'
        
        logger.info(f"Initializing DocumentClassifier on {self.device}")
        self._load_model()
        logger.info("DocumentClassifier initialized successfully")
    
    def _load_model(self):
        """Load model and tokenizer."""
        # Load tokenizer
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
        
        # Load base model
        logger.info(f"Loading base model: {self.base_model_name}")
        
        if self.load_in_8bit:
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)
            base_model = AutoModelForCausalLM.from_pretrained(
                self.base_model_name,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True
            )
        else:
            # Use bf16 for numerical stability (must match training dtype)
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
        logger.info(f"Loading LoRA adapters: {self.model_path}")
        self.model = PeftModel.from_pretrained(base_model, self.model_path)
        
        if self.device == 'cpu':
            self.model = self.model.to(self.device)
        
        self.model.eval()
    
    def _format_prompt(self, text: str) -> str:
        """Format document for inference."""
        prompt = f"""<bos><start_of_turn>user
You are a document classification assistant. Classify the following document into exactly ONE subcategory.
Output ONLY valid JSON: {{"label": "<subcategory>"}}

DOCUMENT:
{text}

Classify this document.<end_of_turn>
<start_of_turn>model
"""
        return prompt
    
    def _format_few_shot_prompt(
        self,
        text: str,
        examples: List[Dict[str, str]]
    ) -> str:
        """Format few-shot prompt for new categories."""
        examples_str = ""
        for i, ex in enumerate(examples, 1):
            examples_str += f"\nExample {i}:\n"
            examples_str += f"Document: {ex['document'][:500]}...\n"
            examples_str += f'Classification: {{"label": "{ex["label"]}"}}\n'
        
        prompt = f"""<bos><start_of_turn>user
You are a document classification assistant. Learn from the examples below and classify the new document.
Output ONLY valid JSON: {{"label": "<subcategory>"}}

EXAMPLES:
{examples_str}
---

NEW DOCUMENT TO CLASSIFY:
{text}

Classify this document.<end_of_turn>
<start_of_turn>model
"""
        return prompt
    
    def _parse_prediction(self, text: str) -> Tuple[Optional[str], bool]:
        """
        Parse model output to extract label.
        
        Returns:
            Tuple of (label, is_valid_json)
        """
        # Try JSON pattern
        pattern = r'\{\s*"label"\s*:\s*"([^"]+)"\s*\}'
        match = re.search(pattern, text)
        
        if match:
            return match.group(1), True
        
        # Try direct JSON parse
        try:
            data = json.loads(text.strip())
            if isinstance(data, dict) and 'label' in data:
                return data['label'], True
        except json.JSONDecodeError:
            pass
        
        # Fallback pattern
        pattern2 = r'label["\s:]+([A-Za-z][A-Za-z0-9_]+)'
        match2 = re.search(pattern2, text, re.IGNORECASE)
        if match2:
            return match2.group(1), False
        
        return None, False
    
    def predict(
        self,
        text: str,
        categories: Optional[List[str]] = None,
        max_new_tokens: int = 50,
        return_raw: bool = False
    ) -> Dict:
        """
        Classify a single document.
        
        Args:
            text: Document text
            categories: List of valid categories (recommended for accuracy)
            max_new_tokens: Maximum tokens to generate
            return_raw: Include raw model output
            
        Returns:
            Dict with 'label' and optionally 'raw_output', 'valid_json'
        """
        # Use proper prompt format with categories if provided
        if categories:
            from training.prompts import format_inference_prompt
            prompt = format_inference_prompt(text, categories)
        else:
            prompt = self._format_prompt(text)
        
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
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode
        generated = outputs[0][inputs['input_ids'].shape[1]:]
        raw_output = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
        
        # Parse
        label, valid_json = self._parse_prediction(raw_output)
        
        result = {
            'label': label or 'UNKNOWN'
        }
        
        if return_raw:
            result['raw_output'] = raw_output
            result['valid_json'] = valid_json
        
        return result
    
    def predict_batch(
        self,
        texts: List[str],
        max_new_tokens: int = 50,
        show_progress: bool = True
    ) -> List[Dict]:
        """
        Classify multiple documents.
        
        Args:
            texts: List of document texts
            max_new_tokens: Maximum tokens to generate
            show_progress: Show progress bar
            
        Returns:
            List of prediction dicts
        """
        from tqdm import tqdm
        
        results = []
        iterator = tqdm(texts, desc="Classifying") if show_progress else texts
        
        for text in iterator:
            result = self.predict(text, max_new_tokens)
            results.append(result)
        
        return results
    
    def predict_few_shot(
        self,
        text: str,
        examples: List[Dict[str, str]],
        max_new_tokens: int = 50
    ) -> Dict:
        """
        Classify with few-shot examples (for new categories).
        
        Args:
            text: Document text to classify
            examples: List of dicts with 'document' and 'label' keys
            max_new_tokens: Maximum tokens to generate
            
        Returns:
            Dict with 'label' key
        """
        # Format few-shot prompt
        prompt = self._format_few_shot_prompt(text, examples)
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
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode
        generated = outputs[0][inputs['input_ids'].shape[1]:]
        raw_output = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
        
        # Parse
        label, _ = self._parse_prediction(raw_output)
        
        return {
            'label': label or 'UNKNOWN',
            'raw_output': raw_output
        }


# Singleton instance for API
_classifier_instance: Optional[DocumentClassifier] = None


def get_classifier(
    model_path: str = "output/lora_adapters/final",
    base_model_name: str = "unsloth/gemma-3-270m",
    force_reload: bool = False
) -> DocumentClassifier:
    """
    Get or create classifier singleton.
    
    Args:
        model_path: Path to LoRA adapters
        base_model_name: Base model name
        force_reload: Force reload model
        
    Returns:
        DocumentClassifier instance
    """
    global _classifier_instance
    
    if _classifier_instance is None or force_reload:
        _classifier_instance = DocumentClassifier(
            model_path=model_path,
            base_model_name=base_model_name
        )
    
    return _classifier_instance
