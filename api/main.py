"""
FastAPI application for the book cover genre classifier.
Exposes: prediction, bulk upload for retraining, retraining trigger, and model status.
"""

import os
import sys
import shutil
import time
from datetime import datetime

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Allow imports from src/ (one level up from api/)
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.prediction import predict_genre
from src.model import retrain_model, load_trained_model

app = FastAPI(title="Book Genre Classifier API")

# Allow requests from any origin (needed so your UI, hosted separately, can call this API)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

START_TIME = time.time()
UPLOAD_DIR = "uploaded_retrain_data"


@app.get("/")
def root():
    return {"message": "Book Genre Classifier API is running"}


@app.get("/health")
def health_check():
    """Model up-time and status — used by the UI's monitoring dashboard."""
    uptime_seconds = time.time() - START_TIME
    try:
        model, class_names = load_trained_model()
        model_loaded = True
    except Exception:
        model_loaded = False
        class_names = []

    return {
        "status": "healthy" if model_loaded else "model unavailable",
        "uptime_seconds": round(uptime_seconds, 2),
        "model_loaded": model_loaded,
        "classes": class_names,
        "checked_at": datetime.utcnow().isoformat()
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Accepts a single book cover image and returns the predicted genre.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        contents = await file.read()
        from io import BytesIO
        result = predict_genre(BytesIO(contents))
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/upload-retrain-data")
async def upload_retrain_data(category: str, files: list[UploadFile] = File(...)):
    """
    Accepts multiple images for a given category, to be used in retraining.
    `category` must match one of the existing class names (e.g. 'Childrens').
    """
    category_dir = os.path.join(UPLOAD_DIR, category)
    os.makedirs(category_dir, exist_ok=True)

    saved_count = 0
    for file in files:
        if not file.content_type.startswith("image/"):
            continue
        file_path = os.path.join(category_dir, file.filename)
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        saved_count += 1

    return {
        "category": category,
        "images_saved": saved_count,
        "message": f"{saved_count} images saved for retraining under '{category}'"
    }


@app.post("/retrain")
def trigger_retrain(epochs: int = 10):
    if not os.path.exists(UPLOAD_DIR) or not os.listdir(UPLOAD_DIR):
        raise HTTPException(status_code=400, detail="No retraining data uploaded yet")

    try:
        model, history = retrain_model(UPLOAD_DIR, epochs=epochs, fine_tune=True)
        final_accuracy = history.history["accuracy"][-1]
        final_val_accuracy = history.history.get("val_accuracy", [None])[-1]

        return {
            "message": "Retraining complete, model updated",
            "epochs_run": len(history.history["accuracy"]),
            "final_train_accuracy": round(final_accuracy, 4),
            "final_val_accuracy": round(final_val_accuracy, 4) if final_val_accuracy is not None else "N/A (dataset too small for validation split)"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retraining failed: {str(e)}")