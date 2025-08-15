from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import tempfile
import os
from utils import predict_audio, load_optimized_model, ASTFeatureExtractorHamming
from contextlib import asynccontextmanager

model = None
feature_extractor = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, feature_extractor
    CHECKPOINT_PATH = os.getenv("CHECKPOINT_PATH", "checkpoint-63480")
    model = load_optimized_model(CHECKPOINT_PATH)
    feature_extractor = ASTFeatureExtractorHamming.from_pretrained(CHECKPOINT_PATH)
    yield

app = FastAPI(
    title="Audio Classification API",
    description="API for audio classification using AST model",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # change to extension URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictionResponse(BaseModel):
    label: str
    confidence: float
    probabilities: dict[str, float]
    timing: dict[str, float]

@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            temp_file.write(file.file.read())
            temp_file_path = temp_file.name

        result = predict_audio(temp_file_path, model, feature_extractor)

        return result   
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)