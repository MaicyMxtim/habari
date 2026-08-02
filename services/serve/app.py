"""Serving API for the fine-tuned AfriBERTa news classifier.

Loads the checkpoint produced by train_transformer.py and exposes a
predict endpoint plus Prometheus metrics, following the same serving
pattern as the Akili platform's house price service.
"""

import os
from pathlib import Path

import torch
from fastapi import FastAPI
from pydantic import BaseModel, Field
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_DIR = Path(os.environ.get("MODEL_DIR", "models/afriberta"))
MAX_LENGTH = 256

torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "2")))

app = FastAPI(title="habari-serve")

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()
labels = [model.config.id2label[i] for i in range(model.config.num_labels)]

try:
    from prometheus_fastapi_instrumentator import Instrumentator

    Instrumentator().instrument(app).expose(app)
except ImportError:
    pass


class PredictRequest(BaseModel):
    text: str = Field(min_length=1, max_length=20000)


class PredictResponse(BaseModel):
    label: str
    confidence: float
    probabilities: dict[str, float]


@app.get("/healthz")
def healthz():
    return {"status": "ok", "model": str(MODEL_DIR), "labels": labels}


@app.post("/predict", response_model=PredictResponse)
@torch.no_grad()
def predict(request: PredictRequest):
    inputs = tokenizer(
        request.text, truncation=True, max_length=MAX_LENGTH, return_tensors="pt"
    )
    probabilities = torch.softmax(model(**inputs).logits[0], dim=-1)
    best = int(probabilities.argmax())
    return PredictResponse(
        label=labels[best],
        confidence=float(probabilities[best]),
        probabilities={label: float(p) for label, p in zip(labels, probabilities)},
    )
