"""
Prompt Templates for Document Classification
=============================================
Centralized prompt definitions for training and inference.
"""

# System instruction for classification task
SYSTEM_PROMPT = """You are a document classification assistant. Your task is to classify documents into exactly ONE subcategory.

RULES:
1. Read the document carefully
2. Choose exactly ONE subcategory from the valid list
3. Output ONLY valid JSON in this exact format: {"label": "<subcategory>"}
4. Do not include any explanations, comments, or additional text
5. The label must exactly match one of the valid subcategories"""

# Full training prompt with categories list
TRAINING_TEMPLATE = """<bos><start_of_turn>user
{system_prompt}

Valid subcategories:
{categories}

---
DOCUMENT:
{document}
---

Classify this document into exactly one subcategory.<end_of_turn>
<start_of_turn>model
{response}<end_of_turn>"""

# Inference prompt (shorter, for production) - legacy, without categories
INFERENCE_TEMPLATE_SIMPLE = """<bos><start_of_turn>user
You are a document classification assistant. Classify the following document into exactly ONE subcategory.
Output ONLY valid JSON: {{"label": "<subcategory>"}}

DOCUMENT:
{document}

Classify this document.<end_of_turn>
<start_of_turn>model
"""

# Inference prompt WITH categories (matches training format - use this!)
INFERENCE_TEMPLATE = """<bos><start_of_turn>user
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

# Few-shot template for new categories (preserves original capability)
FEW_SHOT_TEMPLATE = """<bos><start_of_turn>user
You are a document classification assistant. Learn from the examples and classify the new document.
Output ONLY valid JSON: {{"label": "<subcategory>"}}

{examples}

---
NEW DOCUMENT TO CLASSIFY:
{document}
---

Classify this document.<end_of_turn>
<start_of_turn>model
"""


def format_training_prompt(document: str, categories: list, response: str = "") -> str:
    """
    Format a training example with Gemma-3 chat template.
    
    Args:
        document: Document text content
        categories: List of valid category names
        response: Expected JSON response (for training)
        
    Returns:
        Formatted prompt string
    """
    categories_str = ", ".join(sorted(categories))
    
    return TRAINING_TEMPLATE.format(
        system_prompt=SYSTEM_PROMPT,
        categories=categories_str,
        document=document,
        response=response
    )


def format_inference_prompt(document: str, categories: list = None) -> str:
    """
    Format an inference prompt (production use).
    
    Args:
        document: Document text content
        categories: List of valid category names (REQUIRED for accurate predictions)
        
    Returns:
        Formatted prompt string
    """
    if categories:
        categories_str = ", ".join(sorted(categories))
        return INFERENCE_TEMPLATE.format(
            system_prompt=SYSTEM_PROMPT,
            categories=categories_str,
            document=document
        )
    else:
        # Legacy fallback (will likely produce poor results!)
        return INFERENCE_TEMPLATE_SIMPLE.format(document=document)


def format_few_shot_prompt(document: str, examples: list) -> str:
    """
    Format a few-shot prompt for new categories.
    
    Args:
        document: Document text to classify
        examples: List of dicts with 'document' and 'label' keys
        
    Returns:
        Formatted prompt string
    """
    examples_str = ""
    for i, ex in enumerate(examples, 1):
        examples_str += f"Example {i}:\n"
        examples_str += f"Document: {ex['document'][:500]}...\n"
        examples_str += f"Classification: {{\"label\": \"{ex['label']}\"}}\n\n"
    
    return FEW_SHOT_TEMPLATE.format(
        examples=examples_str.strip(),
        document=document
    )
