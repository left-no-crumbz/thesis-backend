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
    
    # In Cloud Run, files are in the /app directory
    CHECKPOINT_PATH = os.getenv("CHECKPOINT_PATH", "checkpoint-266490")
    
    # Ensure the checkpoint path is an absolute path
    if not os.path.isabs(CHECKPOINT_PATH):
        CHECKPOINT_PATH = os.path.join(os.getcwd(), CHECKPOINT_PATH)
    
    print(f"Loading model from checkpoint: {CHECKPOINT_PATH}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Directory contents: {os.listdir(os.path.dirname(CHECKPOINT_PATH))}")

    try:
        model = load_optimized_model(CHECKPOINT_PATH)
        feature_extractor = ASTFeatureExtractorHamming.from_pretrained(
            CHECKPOINT_PATH,
            local_files_only=True
        )
        print("Model and feature extractor loaded successfully")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        # Don't raise here to allow the app to start even if model loading fails
        # This is important for health checks to pass during deployment
        model = None
        feature_extractor = None
        
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
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            print(f"Removing temporary file: {temp_file_path}")
            os.remove(temp_file_path)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)