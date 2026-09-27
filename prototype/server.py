"""
TeleGuard AI: Interactive Demo Server (FastAPI)
Provides live dashboard for Ideathon presentations and technical mentoring reviews.
"""

import os
import sys
import numpy as np
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Ensure root workspace is on python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from prototype.core.stream_simulator import TelecomCallPipeline
from prototype.run_demo import generate_human_voice_sample, generate_synthetic_voice_sample

app = FastAPI(title="TeleGuard AI In-Network Voice Shield")

pipeline = TelecomCallPipeline(subscriber_id="+37491401122")

class ProcessFrameRequest(BaseModel):
    scenario: str
    frame: int

class ProcessLiveAudioRequest(BaseModel):
    samples: list[float]
    response_latency_ms: float = 220.0

@app.post("/api/process-live-audio")
async def process_live_audio(req: ProcessLiveAudioRequest):
    samples_np = np.array(req.samples, dtype=np.float32)
    # Resample or pad/slice to 400ms window (6400 samples at 16kHz)
    if len(samples_np) > 0:
        result = pipeline.process_incoming_rtp_chunk(samples_np, response_latency_ms=req.response_latency_ms)
    else:
        result = {"error": "Empty audio buffer"}
    return result

@app.get("/api/certificate/status")
async def get_cert_status():
    cert_info = pipeline.certificate_engine.get_current_certificate()
    return {
        "subscriber_id": cert_info["subscriber_id"],
        "current_epoch": cert_info["epoch"],
        "seconds_remaining_in_epoch": cert_info["seconds_remaining"],
        "certificate_fingerprint": cert_info["certificate_fingerprint"]
    }

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    index_path = os.path.join(os.path.dirname(__file__), "static", "index.html")
    return FileResponse(index_path)

@app.post("/api/process-frame")
async def process_frame(req: ProcessFrameRequest):
    chunk_samples = 6400 # 400ms at 16kHz
    call_id = "CALL_DEMO_2026_SESSION"
    now = time.time()
    
    is_registered_claim = False
    
    if req.scenario == "registered_genuine":
        # Sending side: Grandson speaks, device embeds active 30s voice certificate
        raw_voice = generate_human_voice_sample(duration_sec=0.4)
        pcm_chunk = pipeline.voice_cert_embedder.embed_certificate(raw_voice, call_id=call_id, timestamp=now)
        response_latency = 190.0
        is_registered_claim = True
    elif req.scenario == "registered_imposter_ai":
        # Attacker: Uses ElevenLabs/XTTS to clone Grandson's voice, but lacks secret key!
        pcm_chunk = generate_synthetic_voice_sample(duration_sec=0.4)
        response_latency = 1100.0
        is_registered_claim = True
    elif req.scenario == "registered_replay":
        # Attacker: Replays a recorded call from 5 minutes ago (expired epoch)
        raw_voice = generate_human_voice_sample(duration_sec=0.4)
        old_epoch_time = now - 180 # 6 epochs ago
        pcm_chunk = pipeline.voice_cert_embedder.embed_certificate(raw_voice, call_id=call_id, timestamp=old_epoch_time)
        response_latency = 210.0
        is_registered_claim = True
    elif req.scenario == "human":
        # Unregistered regular human call
        pcm_chunk = generate_human_voice_sample(duration_sec=0.4)
        response_latency = 220.0
    elif req.scenario == "synthetic_opensource":
        # Unregistered open-source synthetic call (XTTS)
        pcm_chunk = generate_synthetic_voice_sample(duration_sec=0.4)
        response_latency = 1180.0
    else: # commercial watermarked
        pcm_chunk = generate_synthetic_voice_sample(duration_sec=0.4)
        sig = pipeline.watermark_detector.signatures["SynthID_Audio"]
        pcm_chunk[-len(sig):] += sig * 0.4
        response_latency = 1250.0

    result = pipeline.process_incoming_rtp_chunk(
        pcm_chunk, 
        response_latency_ms=response_latency,
        is_registered_caller_claim=is_registered_claim,
        call_id=call_id
    )
    return result

if __name__ == "__main__":
    import uvicorn
    print("\n" + "=" * 70)
    print(" TeleGuard AI Demo Server running at: http://127.0.0.1:8000")
    print(" Open the browser to interact with the live telecom SBC stream demo.")
    print("=" * 70 + "\n")
    uvicorn.run(app, host="127.0.0.1", port=8000)
