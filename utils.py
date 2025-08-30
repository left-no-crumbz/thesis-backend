from functools import wraps
import time
from typing import Dict, Any, Callable, TypeVar, Optional
from optimum.onnxruntime import ORTModelForAudioClassification
from feature_extractor import ASTFeatureExtractorHamming
import torch
import torchaudio
import numpy as np
import onnxruntime as ort
import os
from typing import Tuple
from functools import lru_cache

def time_execution(metric_name: str):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            elapsed = time.time() - start_time
            
            if not hasattr(wrapper, 'timings'):
                wrapper.timings = {}
            wrapper.timings[metric_name] = elapsed
            
            if isinstance(result, dict):
                if 'timing' not in result:
                    result['timing'] = {}
                result['timing'][metric_name] = elapsed
                
            return result
        return wrapper
    return decorator

def get_session_options():
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.enable_mem_pattern = True
    options.enable_mem_reuse = True
    options.enable_cpu_mem_arena = True
    return options

def get_providers():
    return [
        (
            'CPUExecutionProvider',
            {
                'arena_extend_strategy': 'kSameAsRequested',
                'intra_op_num_threads': os.cpu_count(),
            }
        )
    ]

@time_execution('model_loading')
@lru_cache(maxsize=1)
def load_optimized_model(checkpoint_path: str):
    session_options = get_session_options()
    providers = get_providers()
    
    print(f"[DEBUG] Attempting to load model from: {checkpoint_path}")
    
    # Check if path exists and is a directory
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Model directory not found at {checkpoint_path}")
    
    if not os.path.isdir(checkpoint_path):
        checkpoint_path = os.path.dirname(checkpoint_path)
    
    print(f"[DEBUG] Model directory contents: {os.listdir(checkpoint_path)}")

    # Ensure ONNX file exists to avoid falling back to Hub logic
    onnx_path = os.path.join(checkpoint_path, "model.onnx")
    if not os.path.exists(onnx_path):
        raise FileNotFoundError(
            f"Expected ONNX file not found at {onnx_path}. Ensure the directory with model.onnx is packaged into the image."
        )
    
    try:
        model = ORTModelForAudioClassification.from_pretrained(
            checkpoint_path,
            local_files_only=True,
            provider=providers[0][0] if providers else "CPUExecutionProvider",
            provider_options=dict(providers[0][1]) if providers else None,
            session_options=session_options,
        )
        print("[DEBUG] Model loaded successfully")
        return model
    except Exception as e:
        print(f"[ERROR] Failed to load model: {str(e)}")
        raise

@time_execution('audio_loading')
def load_audio_torchaudio(file_path: str, target_sr: int) -> np.ndarray:
    waveform, sample_rate = torchaudio.load(file_path)
    
    if waveform.dim() > 1 and waveform.size(0) > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    if sample_rate != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sr)
        waveform = resampler(waveform)
    
    return waveform.squeeze().numpy().astype(np.float32)

@time_execution('feature_extraction')
def extract_features(audio: np.ndarray, feature_extractor: ASTFeatureExtractorHamming):
    return feature_extractor(
        audio,
        sampling_rate=feature_extractor.sampling_rate,
        padding=True,
        return_tensors="pt",
        return_attention_mask=True,
    )

@time_execution('inference')
def run_inference(model: ORTModelForAudioClassification, inputs: Dict[str, torch.Tensor]):
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        return torch.softmax(logits, dim=-1)

def predict_audio(file_path: str, model: ORTModelForAudioClassification, feature_extractor: ASTFeatureExtractorHamming):
    try:
        audio = load_audio_torchaudio(file_path, feature_extractor.sampling_rate)
    except Exception as e:
        print(f"Error loading audio with torchaudio: {e}")
        print("Falling back to librosa...")
        import librosa
        audio, _ = librosa.load(file_path, sr=feature_extractor.sampling_rate)
    
    inputs = extract_features(audio, feature_extractor)
    probabilities = run_inference(model, inputs)
    
    predicted_class = torch.argmax(probabilities, dim=1).item()
    confidence = probabilities[0][predicted_class].item()
    
    timing = {}
    for func in [load_audio_torchaudio, extract_features, run_inference, load_optimized_model]:
        if hasattr(func, 'timings'):
            timing.update(func.timings)
    
    timing['total_seconds'] = sum(timing.values())
    
    return {
        "label": model.config.id2label.get(predicted_class, str(predicted_class)),
        "confidence": confidence,
        "probabilities": {model.config.id2label[i]: prob.item() for i, prob in enumerate(probabilities[0])},
        "timing": timing
    }