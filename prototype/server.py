"""
TeleGuard AI: Real-Time AI Voice Detection Server
Focused 100% on detecting AI-Generated Speech vs Real Human Speech.
"""

import os
import sys
import time
import numpy as np
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel

# Ensure root workspace is on python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from prototype.core.robust_detector import RealtimeVoiceDetector
from prototype.run_demo import generate_human_voice_sample, generate_synthetic_voice_sample

app = FastAPI(title="TeleGuard AI: Real-Time AI Voice Detection")

detector = RealtimeVoiceDetector(target_sample_rate=16000)

class ProcessLiveAudioRequest(BaseModel):
    samples: list[float]
    client_sample_rate: int = 16000

class ProcessFrameRequest(BaseModel):
    scenario: str
    frame: int

def resample_to_16k(audio: np.ndarray, orig_sr: int) -> np.ndarray:
    """Downsamples client audio (44.1k/48k) to 16kHz using linear interpolation."""
    if orig_sr == 16000 or len(audio) == 0:
        return audio
    target_len = int(len(audio) * 16000 / orig_sr)
    if target_len < 10:
        return audio
    orig_indices = np.arange(len(audio))
    target_indices = np.linspace(0, len(audio) - 1, target_len)
    return np.interp(target_indices, orig_indices, audio).astype(np.float32)

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    index_path = os.path.join(os.path.dirname(__file__), "static", "index.html")
    return FileResponse(index_path)

@app.post("/api/process-live-audio")
async def process_live_audio(req: ProcessLiveAudioRequest):
    start_t = time.perf_counter()
    raw_samples = np.array(req.samples, dtype=np.float32)
    
    # Resample from client's browser rate (e.g. 48kHz) to 16kHz
    audio_16k = resample_to_16k(raw_samples, req.client_sample_rate)
    
    result = detector.analyze_audio(audio_16k, sample_rate=16000)
    elapsed_ms = (time.perf_counter() - start_t) * 1000.0

    return {
        "processing_latency_ms": round(elapsed_ms, 2),
        "synthetic_probability": result["synthetic_probability"],
        "threat_level": result["threat_level"],
        "verdict": result["verdict"],
        "is_synthetic": result["is_synthetic"],
        "details": result["details"],
        "alert_delivered": {
            "in_call_whisper": result["is_synthetic"],
            "flash_sms": "[SECURITY WARNING] AI-Generated Synthetic Voice Detected!" if result["is_synthetic"] else None
        }
    }

@app.post("/api/process-frame")
async def process_frame(req: ProcessFrameRequest):
    start_t = time.perf_counter()
    fs = 16000
    
    if req.scenario == "human":
        # Authentic human speech with biological jitter
        t = np.linspace(0, 0.4, int(fs * 0.4))
        f0 = 135.0 * (1.0 + 0.018 * np.sin(2 * np.pi * 5.5 * t) + np.random.normal(0, 0.005, len(t)))
        phase = 2 * np.pi * np.cumsum(f0) / fs
        pcm_chunk = np.sin(phase) + 0.5 * np.sin(2 * phase) + 0.25 * np.sin(3 * phase)
    else:
        # AI cloned speech: perfectly static neural pitch trajectory
        t = np.linspace(0, 0.4, int(fs * 0.4))
        f0 = 135.0
        pcm_chunk = np.sin(2 * np.pi * f0 * t) + 0.5 * np.sin(2 * np.pi * 2 * f0 * t) + 0.25 * np.sin(2 * np.pi * 3 * f0 * t)

    result = detector.analyze_audio(pcm_chunk, sample_rate=fs)
    elapsed_ms = (time.perf_counter() - start_t) * 1000.0

    return {
        "frame_index": req.frame,
        "processing_latency_ms": round(elapsed_ms, 2),
        "synthetic_probability": result["synthetic_probability"],
        "threat_level": result["threat_level"],
        "verdict": result["verdict"],
        "is_synthetic": result["is_synthetic"],
        "details": result["details"],
        "alert_delivered": {
            "in_call_whisper": result["is_synthetic"],
            "flash_sms": "[SECURITY WARNING] AI-Generated Synthetic Voice Detected!" if result["is_synthetic"] else None
        }
    }

@app.post("/api/detect-audio-file")
async def detect_audio_file(file: UploadFile = File(...)):
    """Upload any audio file to test AI detection."""
    content = await file.read()
    # Read raw bytes as 16-bit PCM if wav, or float
    try:
        audio = np.frombuffer(content, dtype=np.int16).astype(np.float32) / 32768.0
        if len(audio) < 1600:
            return {"error": "Audio file too short (minimum 0.2s required)"}
        result = detector.analyze_audio(audio[:32000], sample_rate=16000)
        return {
            "filename": file.filename,
            "verdict": result["verdict"],
            "synthetic_probability": result["synthetic_probability"],
            "threat_level": result["threat_level"],
            "details": result["details"]
        }
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    print("\n" + "=" * 70)
    print(" TeleGuard AI: Real-Time AI Voice Detection Server")
    print(" Listening at: http://127.0.0.1:8000")
    print("=" * 70 + "\n")
    uvicorn.run(app, host="127.0.0.1", port=8000)
