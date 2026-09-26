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

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    index_path = os.path.join(os.path.dirname(__file__), "static", "index.html")
    return FileResponse(index_path)

@app.post("/api/process-frame")
async def process_frame(req: ProcessFrameRequest):
    chunk_samples = 6400 # 400ms at 16kHz
    
    if req.scenario == "human":
        # Generate authentic human speech with biological micro-jitter
        pcm_chunk = generate_human_voice_sample(duration_sec=0.4)
        response_latency = 220.0
    elif req.scenario == "synthetic_opensource":
        # Generate open-source model speech with vocoder upsampling artifacts
        pcm_chunk = generate_synthetic_voice_sample(duration_sec=0.4)
        response_latency = 1180.0
    else: # commercial watermarked
        pcm_chunk = generate_synthetic_voice_sample(duration_sec=0.4)
        # Add SynthID spread-spectrum watermark signature in high frequencies
        sig = pipeline.watermark_detector.signatures["SynthID_Audio"]
        pcm_chunk[-len(sig):] += sig * 0.4
        response_latency = 1250.0

    result = pipeline.process_incoming_rtp_chunk(pcm_chunk, response_latency_ms=response_latency)
    return result

if __name__ == "__main__":
    import uvicorn
    print("\n" + "=" * 70)
    print(" TeleGuard AI Demo Server running at: http://127.0.0.1:8000")
    print(" Open the browser to interact with the live telecom SBC stream demo.")
    print("=" * 70 + "\n")
    uvicorn.run(app, host="127.0.0.1", port=8000)
