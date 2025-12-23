#!/usr/bin/env python3
"""
FastAPI Application for Document Classification
================================================
RESTful API for classifying documents using fine-tuned Gemma-3.

Usage:
    # Start server
    uvicorn inference.api:app --host 0.0.0.0 --port 8000
    
    # With auto-reload for development
    uvicorn inference.api:app --host 0.0.0.0 --port 8000 --reload

Endpoints:
    POST /predict         - Classify single document
    POST /predict_batch   - Classify multiple documents
    POST /predict_fewshot - Few-shot classification
    GET  /health          - Health check
    GET  /info            - Model info
"""

import os
import sys
import json
import logging
import time
from pathlib import Path
from typing import List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global classifier instance
classifier = None


# =============================================================================
# Configuration
# =============================================================================

# Load from environment or use defaults
MODEL_PATH = os.environ.get("MODEL_PATH", "output/lora_adapters/final")
BASE_MODEL = os.environ.get("BASE_MODEL", "unsloth/gemma-3-270m")
LOAD_ON_STARTUP = os.environ.get("LOAD_ON_STARTUP", "true").lower() == "true"


# =============================================================================
# Request/Response Models
# =============================================================================

class PredictRequest(BaseModel):
    """Request for single document classification."""
    text: str = Field(..., description="Document text to classify", min_length=1)
    categories: Optional[List[str]] = Field(None, description="List of valid categories (recommended for accurate prediction)")
    return_raw: bool = Field(False, description="Include raw model output")
    
    class Config:
        json_schema_extra = {
            "example": {
                "text": "This is an offer letter confirming your employment...",
                "categories": ["OfferLetter", "SalaryStructure", "EmploymentContract"],
                "return_raw": False
            }
        }


class PredictResponse(BaseModel):
    """Response for single document classification."""
    label: str = Field(..., description="Predicted subcategory")
    raw_output: Optional[str] = Field(None, description="Raw model output (if requested)")
    valid_json: Optional[bool] = Field(None, description="Whether output was valid JSON")
    
    class Config:
        json_schema_extra = {
            "example": {
                "label": "OfferLetter"
            }
        }


class BatchPredictRequest(BaseModel):
    """Request for batch document classification."""
    texts: List[str] = Field(..., description="List of document texts", min_length=1)
    
    class Config:
        json_schema_extra = {
            "example": {
                "texts": [
                    "This is an offer letter...",
                    "Please find attached your salary structure..."
                ]
            }
        }


class BatchPredictResponse(BaseModel):
    """Response for batch document classification."""
    predictions: List[PredictResponse] = Field(..., description="List of predictions")
    count: int = Field(..., description="Number of documents classified")


class FewShotExample(BaseModel):
    """A single example for few-shot learning."""
    document: str = Field(..., description="Example document text")
    label: str = Field(..., description="Example label")


class FewShotRequest(BaseModel):
    """Request for few-shot classification."""
    text: str = Field(..., description="Document to classify")
    examples: List[FewShotExample] = Field(..., description="Few-shot examples", min_length=1)
    
    class Config:
        json_schema_extra = {
            "example": {
                "text": "Document to classify...",
                "examples": [
                    {"document": "Example doc 1...", "label": "Category1"},
                    {"document": "Example doc 2...", "label": "Category2"}
                ]
            }
        }


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    device: Optional[str] = None


class InfoResponse(BaseModel):
    """Model info response."""
    model_path: str
    base_model: str
    device: str
    status: str


class ChatRequest(BaseModel):
    """Request for chat interaction."""
    message: str = Field(..., description="User message", min_length=1)
    use_lora: bool = Field(False, description="Whether to use LoRA adapters (classification mode)")
    max_new_tokens: int = Field(256, description="Maximum tokens to generate")
    
    class Config:
        json_schema_extra = {
            "example": {
                "message": "What is machine learning?",
                "use_lora": False,
                "max_new_tokens": 256
            }
        }


class ChatResponse(BaseModel):
    """Response for chat interaction."""
    response: str = Field(..., description="Model response")
    mode: str = Field(..., description="Mode used (base or lora)")


# =============================================================================
# Lifespan Management
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle startup and shutdown."""
    global classifier
    
    # Startup
    if LOAD_ON_STARTUP:
        logger.info("Loading model on startup...")
        try:
            from inference.predictor import DocumentClassifier
            classifier = DocumentClassifier(
                model_path=MODEL_PATH,
                base_model_name=BASE_MODEL
            )
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            logger.warning("API starting without model. Use /reload to load later.")
    else:
        logger.info("Model loading deferred. Use /reload endpoint to load.")
    
    yield
    
    # Shutdown
    logger.info("Shutting down...")
    classifier = None


# =============================================================================
# FastAPI App
# =============================================================================

app = FastAPI(
    title="Document Classification API",
    description="Classify documents into subcategories using fine-tuned Gemma-3",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# Helper Functions
# =============================================================================

def ensure_model_loaded():
    """Check if model is loaded, raise error if not."""
    if classifier is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please wait for initialization or call /reload"
        )


# =============================================================================
# Endpoints
# =============================================================================

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Check API health and model status."""
    return HealthResponse(
        status="healthy",
        model_loaded=classifier is not None,
        device=classifier.device if classifier else None
    )


@app.get("/info", response_model=InfoResponse, tags=["Health"])
async def get_info():
    """Get model and API information."""
    ensure_model_loaded()
    return InfoResponse(
        model_path=MODEL_PATH,
        base_model=BASE_MODEL,
        device=classifier.device,
        status="ready"
    )


@app.post("/reload", tags=["Health"])
async def reload_model(
    model_path: Optional[str] = None,
    base_model: Optional[str] = None
):
    """Reload model (optionally with new paths)."""
    global classifier, MODEL_PATH, BASE_MODEL
    
    if model_path:
        MODEL_PATH = model_path
    if base_model:
        BASE_MODEL = base_model
    
    try:
        from inference.predictor import DocumentClassifier
        classifier = DocumentClassifier(
            model_path=MODEL_PATH,
            base_model_name=BASE_MODEL
        )
        return {"status": "success", "message": "Model reloaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to reload model: {e}")


@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
async def predict(request: PredictRequest):
    """
    Classify a single document.
    
    Returns the predicted subcategory label as JSON.
    """
    ensure_model_loaded()
    
    start_time = time.time()
    
    try:
        result = classifier.predict(
            text=request.text,
            categories=request.categories,
            return_raw=request.return_raw
        )
        
        elapsed = time.time() - start_time
        logger.info(f"Prediction completed in {elapsed:.2f}s: {result['label']}")
        
        return PredictResponse(**result)
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict_batch", response_model=BatchPredictResponse, tags=["Prediction"])
async def predict_batch(request: BatchPredictRequest):
    """
    Classify multiple documents.
    
    More efficient for processing large numbers of documents.
    """
    ensure_model_loaded()
    
    if len(request.texts) > 100:
        raise HTTPException(
            status_code=400,
            detail="Maximum 100 documents per batch"
        )
    
    start_time = time.time()
    
    try:
        results = classifier.predict_batch(
            texts=request.texts,
            show_progress=False
        )
        
        elapsed = time.time() - start_time
        logger.info(f"Batch prediction ({len(request.texts)} docs) completed in {elapsed:.2f}s")
        
        return BatchPredictResponse(
            predictions=[PredictResponse(**r) for r in results],
            count=len(results)
        )
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict_fewshot", response_model=PredictResponse, tags=["Prediction"])
async def predict_few_shot(request: FewShotRequest):
    """
    Classify using few-shot examples.
    
    Use this for classifying into new categories not seen during training.
    Provide examples of the new categories in the request.
    """
    ensure_model_loaded()
    
    start_time = time.time()
    
    try:
        # Convert examples to dict format
        examples = [
            {"document": ex.document, "label": ex.label}
            for ex in request.examples
        ]
        
        result = classifier.predict_few_shot(
            text=request.text,
            examples=examples
        )
        
        elapsed = time.time() - start_time
        logger.info(f"Few-shot prediction completed in {elapsed:.2f}s: {result['label']}")
        
        return PredictResponse(label=result['label'])
        
    except Exception as e:
        logger.error(f"Few-shot prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(request: ChatRequest):
    """
    Chat with the LLM (base model or with LoRA adapters).
    
    This allows general conversation with the model.
    Set use_lora=False to use base model capabilities.
    """
    ensure_model_loaded()
    
    start_time = time.time()
    
    try:
        # Format simple chat prompt
        prompt = f"<bos><start_of_turn>user\n{request.message}<end_of_turn>\n<start_of_turn>model\n"
        
        inputs = classifier.tokenizer(
            prompt,
            return_tensors='pt',
            truncation=True,
            max_length=4096
        ).to(classifier.device)
        
        import torch
        with torch.no_grad():
            outputs = classifier.model.generate(
                **inputs,
                max_new_tokens=request.max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=classifier.tokenizer.pad_token_id,
                eos_token_id=classifier.tokenizer.eos_token_id
            )
        
        # Decode only new tokens
        generated = outputs[0][inputs['input_ids'].shape[1]:]
        response = classifier.tokenizer.decode(generated, skip_special_tokens=True).strip()
        
        elapsed = time.time() - start_time
        logger.info(f"Chat completed in {elapsed:.2f}s")
        
        return ChatResponse(
            response=response,
            mode="lora" if request.use_lora else "base"
        )
        
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/categories", tags=["Info"])
async def get_categories():
    """Get list of supported classification categories."""
    return {
        "categories": [
            "AdmissionForm", "ArticlesOfOrganization", "BusinessAccountPayable", 
            "BusinessBalanceSheet", "BusinessInsurance", "BusinessLicense", 
            "BusinessPayroll", "CertificateOfEmployment", "ConfidentialityAgreement",
            "ConflictOfInterest", "CustomerComplaints", "DataPrivacy", 
            "DischargeSummary", "Diversity", "DoNotResuscitateForm", "DressCode",
            "EmploymentContract", "HIPAA", "Harassment", "Immunization",
            "IndemnityAgreement", "InvestigationReports", "LiabilityWaiver",
            "MeetingMinutes", "MissedAppointmentPolicy", "MissionVisionAndStrategy",
            "OfferLetter", "OperativeReport", "OrganizationChart", "PerformanceAppraisal",
            "PerformanceImprovementPlan", "PoliticalParty", "ProgressSheet",
            "RecordsManagement", "Recruitment", "SalaryStructure", "ShareHolderAgreement",
            "Telecommuting", "TermsOfUse", "TimeOff", "Will"
        ],
        "count": 41
    }


# Serve static UI files
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Mount UI directory if it exists
ui_path = Path(__file__).parent.parent / "ui"
if ui_path.exists():
    app.mount("/ui", StaticFiles(directory=str(ui_path), html=True), name="ui")

@app.get("/", tags=["UI"])
async def root():
    """Redirect to UI or API docs."""
    ui_file = Path(__file__).parent.parent / "ui" / "index.html"
    if ui_file.exists():
        return FileResponse(str(ui_file))
    return {"message": "Document Classification API", "docs": "/docs", "ui": "/ui"}


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")
    
    print(f"Starting server at http://{host}:{port}")
    print(f"UI available at: http://{host}:{port}/")
    print(f"API docs at: http://{host}:{port}/docs")
    
    uvicorn.run(app, host=host, port=port)
