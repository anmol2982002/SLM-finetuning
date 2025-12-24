# How We Built the Gemma-3 Document Classifier

> A beginner-friendly guide to fine-tuning a small language model for document classification

---

## What Is This Project?

We took Google's **Gemma-3 270M** model (a small but capable language model) and taught it to classify documents into **41 different categories** like Offer Letters, Contracts, Salary Structures, etc.

**The Result:** A model that reads any document and tells you exactly what type it is — with **98.7% accuracy**.

---

## The Journey: From Documents to AI Classifier

```
📁 Raw Documents → 📝 Extracted Text → 🧠 Model Training → ✨ Working Classifier
```

### Step 1: Collecting Documents 📁

We gathered real-world documents organized by type:

```
data/
├── OfferLetter/
│   ├── offer_john.pdf
│   └── offer_sarah.docx
├── SalaryStructure/
│   └── compensation_2024.pdf
├── EmploymentContract/
│   └── contract_template.pdf
└── ... (41 different categories)
```

**Each folder name = the category we want the model to learn.**

---

### Step 2: Extracting Text 📝

Documents come in many formats (PDF, Word, etc.). We used **Apache Tika** to extract plain text:

```bash
python scripts/extract_text.py --input_dir data/ --output extracted.jsonl
```

**What this does:**
- Opens each PDF/DOCX file
- Pulls out all the readable text
- Saves it in a simple format the model can understand

---

### Step 3: Creating Training Data 🎓

We converted the extracted text into **instruction-style training examples**. This teaches the model HOW to respond.

#### How Training Data Looks

Each training example has 3 parts:

| Part | Purpose |
|------|---------|
| **Instruction** | Tells the model what to do |
| **Document** | The actual text to classify |
| **Response** | The correct answer (JSON format) |

**Real Example from Our Training Data:**

```
<bos><start_of_turn>user
You are a document classification assistant. Your task is to classify documents into exactly ONE subcategory.

RULES:
1. Read the document carefully
2. Choose exactly ONE subcategory from the valid list
3. Output ONLY valid JSON in this exact format: {"label": "<subcategory>"}

Valid subcategories:
AdmissionForm, BusinessInsurance, ConfidentialityAgreement, EmploymentContract, 
HIPAA, OfferLetter, SalaryStructure, ... (all 41 categories listed)

---
DOCUMENT:
RE: OFFER OF EMPLOYMENT

Dear John Smith,

We are pleased to extend this offer of employment for the position of 
Software Engineer at ABC Technologies Inc.

Start Date: January 15, 2025
Salary: $120,000 per annum
Department: Engineering
Reporting to: VP of Engineering

This offer is contingent upon successful completion of background verification...
---

Classify this document into exactly one subcategory.<end_of_turn>
<start_of_turn>model
{"label": "OfferLetter"}<end_of_turn>
```

**Why this format?**
- The model learns to follow instructions
- It learns to output clean JSON (easy to parse programmatically)
- It learns the relationship between document content and categories

---

### Step 4: Training with LoRA 🧠

Instead of changing the entire Gemma-3 model (which would require massive compute), we used a technique called **LoRA (Low-Rank Adaptation)**.

#### What is LoRA?

Think of it like this:
- **The original model** = A skilled general-purpose assistant
- **LoRA** = A small "addon" that gives it specialized knowledge

| Approach | Parameters Changed | GPU Memory | Training Time |
|----------|-------------------|------------|---------------|
| Full Fine-tuning | 270 million | 16+ GB | Hours |
| **LoRA** | ~330 thousand | 4-8 GB | Minutes |

**LoRA only changes 0.1% of the model while achieving the same results!**

#### Training Command

```bash
python training/train_lora.py --config training/config.yaml
```

#### Key Training Settings

| Setting | Value | Why? |
|---------|-------|------|
| Learning Rate | 0.0002 | Small steps = stable learning |
| Epochs | 3 | See training data 3 times |
| Batch Size | 2 | Fits in limited GPU memory |
| LoRA Rank | 16 | Good balance of size vs capability |

---

### Step 5: Evaluation 📊

After training, we tested the model on documents it had NEVER seen before.

```bash
python evaluation/evaluate.py \
    --model_path output/lora_adapters/final \
    --test_file data/processed/test.jsonl
```

#### Results

| Metric | Score |
|--------|-------|
| **Accuracy** | 98.7% |
| **F1 Score** | 0.986 |
| Perfect Categories (100% correct) | 27 out of 41 |

The model correctly classifies almost every document!

---

## How the Model Makes Predictions

When you give the model a new document:

```
Input: "Dear Candidate, We are pleased to offer you the position of..."

↓ Model processes text ↓

Output: {"label": "OfferLetter"}
```

The model learned patterns like:
- "offer of employment" → OfferLetter
- "salary breakdown" → SalaryStructure  
- "non-disclosure agreement" → ConfidentialityAgreement

---

## Few-Shot Classification: Handling NEW Categories

### What Is Few-Shot?

Few-shot classification lets you classify documents into **categories the model was never trained on** — by providing just 2-5 examples.

### Does It Use the Fine-Tuned Model?

**Yes!** Few-shot uses your LoRA fine-tuned model because:
- It already knows HOW to classify documents
- It already outputs clean JSON format
- The examples just provide **context** for new categories

### The Prompt Structure

When you provide examples, this is what gets sent to the model:

```
<bos><start_of_turn>user
You are a document classification assistant. Learn from the examples and classify the new document.
Output ONLY valid JSON: {"label": "<subcategory>"}

Example 1:
Document: Thank you for your order #12345. Your invoice total is $500...
Classification: {"label": "Invoice"}

Example 2:
Document: Please find attached our quotation for the requested services...
Classification: {"label": "Quote"}

---
NEW DOCUMENT TO CLASSIFY:
We are pleased to provide this price estimate for your consideration...
---

Classify this document.<end_of_turn>
<start_of_turn>model
```

**Model outputs:** `{"label": "Quote"}`

### How It Works (Step by Step)

```
┌─────────────────────────────────────────────────────────────┐
│  1. User provides examples (text + label pairs)             │
│     Example: "Invoice for order #123..." → "Invoice"        │
├─────────────────────────────────────────────────────────────┤
│  2. System builds the prompt                                │
│     - Adds all examples with their labels                   │
│     - Adds the new document to classify                     │
├─────────────────────────────────────────────────────────────┤
│  3. Fine-tuned model processes the prompt                   │
│     - Learns patterns from examples                         │
│     - Applies learned pattern to new document               │
├─────────────────────────────────────────────────────────────┤
│  4. Model outputs JSON                                      │
│     {"label": "Invoice"}                                    │
└─────────────────────────────────────────────────────────────┘
```

### Standard vs Few-Shot: When to Use What?

| Aspect | Standard Classification | Few-Shot Classification |
|--------|------------------------|------------------------|
| **Categories** | Fixed 41 categories | Any custom categories |
| **Prompt** | Lists all 41 categories | Shows 1-5 examples |
| **Use case** | Known document types | NEW document types |
| **Speed** | Faster (shorter prompt) | Slightly slower |

### Example Use Case

**Scenario:** You receive "Purchase Order" documents, but this wasn't in the original 41 categories.

**Solution with Few-Shot:**
1. Provide 2-3 examples of Purchase Orders
2. Label them as "PurchaseOrder"
3. Model learns from examples and classifies new ones correctly!

**Without few-shot:** Model would guess from existing 41 categories (probably wrong)

---

## Key Technologies Used

| Technology | Purpose |
|------------|---------|
| **Gemma-3 270M** | Base language model (Google) |
| **LoRA / PEFT** | Efficient fine-tuning technique |
| **PyTorch** | Deep learning framework |
| **Transformers** | Hugging Face model library |
| **Apache Tika** | Document text extraction |
| **FastAPI** | REST API for serving predictions |

---

## Project Files Overview

```
📁 SLM-Finetuning/
│
├── 📂 scripts/
│   ├── extract_text.py      # Step 2: Extract text from documents
│   └── prepare_dataset.py   # Step 3: Create training data
│
├── 📂 training/
│   ├── config.yaml          # Training settings
│   ├── train_lora.py        # Step 4: Train the model
│   └── prompts.py           # Instruction templates
│
├── 📂 evaluation/
│   └── evaluate.py          # Step 5: Test the model
│
├── 📂 inference/
│   ├── api.py               # REST API server
│   └── predictor.py         # Core prediction logic
│
└── 📂 ui/
    └── index.html           # Web interface
```

---

## Try It Yourself!

### Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Start the API server
python -m uvicorn inference.api:app --port 8000

# 3. Open browser
# Go to http://localhost:8000
```

### Classify a Document

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "This is your offer letter for the position of..."}'

# Response: {"label": "OfferLetter"}
```

---

## Summary

| Step | What We Did | Tool Used |
|------|-------------|-----------|
| 1 | Collected 840+ documents in 41 categories | Manual organization |
| 2 | Extracted text from PDF/DOCX files | Apache Tika |
| 3 | Created instruction-style training data | Custom Python script |
| 4 | Fine-tuned Gemma-3 using LoRA | Hugging Face + PEFT |
| 5 | Evaluated on test set | Custom evaluation script |
| 6 | Deployed as REST API + Web UI | FastAPI |

**Total training time:** ~30 minutes on a single GPU  
**Final model size:** Only 1.3 MB of LoRA weights (original model stays unchanged!)

---

*Built with ❤️ using Google Gemma-3 and Hugging Face PEFT*
