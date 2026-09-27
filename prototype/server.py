"""
TeleGuard AI: Real-Time AI Voice Detection Server
Focused 100% on detecting AI-Generated Speech vs Real Human Speech.
"""

import os
import sys
import time
import io
import wave
import base64
import numpy as np
try:
    import soundfile as sf
    HAVE_SOUNDFILE = True
except ImportError:
    HAVE_SOUNDFILE = False

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Ensure root workspace is on python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from prototype.core.robust_detector import RealtimeVoiceDetector
from prototype.run_demo import generate_human_voice_sample, generate_synthetic_voice_sample

app = FastAPI(title="TeleGuard AI: Real-Time AI Voice Detection")

static_dir = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")

detector = RealtimeVoiceDetector(target_sample_rate=16000)

class AnalyzeRecordingRequest(BaseModel):
    samples: list[float]
    client_sample_rate: int = 16000

class AnalyzePresetRequest(BaseModel):
    preset: str # "human" | "ai_clone"

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

def sanitize_for_json(obj):
    """Recursively converts all numpy scalars, booleans, and arrays to native python types."""
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [sanitize_for_json(x) for x in obj]
    elif isinstance(obj, (np.bool_, np.bool)):
        return bool(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

def pcm_to_wav_base64(pcm_data: np.ndarray, sample_rate: int = 16000) -> str:
    """Encodes float32 PCM samples into a browser-playable WAV Data URL."""
    try:
        buf = io.BytesIO()
        with wave.open(buf, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            int16_data = (np.clip(pcm_data, -1.0, 1.0) * 32767).astype(np.int16)
            wf.writeframes(int16_data.tobytes())
        return "data:audio/wav;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception:
        return ""

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    index_path = os.path.join(os.path.dirname(__file__), "static", "index.html")
    return FileResponse(index_path)

@app.post("/api/analyze-recording")
async def analyze_recording(req: AnalyzeRecordingRequest):
    """
    Analyzes a full recorded voice segment (from Start Recording -> Stop Recording).
    Downsamples to 16kHz, computes pitch micro-jitter, amplitude shimmer,
    and dissertation LPC residual next-bit predictability test.
    """
    start_t = time.perf_counter()
    try:
        raw_samples = np.array(req.samples, dtype=np.float32)
        
        # Resample from client browser rate (44.1k/48k) to 16kHz
        audio_16k = resample_to_16k(raw_samples, req.client_sample_rate)
        
        # Run full recording forensic analysis
        result = detector.analyze_full_recording(audio_16k, sample_rate=16000)
        elapsed_ms = (time.perf_counter() - start_t) * 1000.0

        # Save last recording for forensic audit
        try:
            debug_path = os.path.join(os.path.dirname(__file__), "last_recording.wav")
            if HAVE_SOUNDFILE:
                sf.write(debug_path, audio_16k, 16000)
        except Exception:
            pass

        bio = result.get("biometrics", {})
        print(f"[TeleGuard AI] Recording analyzed: dur={len(audio_16k)/16000:.2f}s | verdict={result.get('verdict')} | prob={result.get('synthetic_probability')} | tremor={bio.get('laryngeal_tremor_pct')}% | comb={bio.get('high_freq_comb_periodicity')} | rap={bio.get('pitch_jitter_rap_pct')}% | latency={elapsed_ms:.1f}ms")
        
        # Generate WAV data URL for immediate audio playback in UI
        audio_url = pcm_to_wav_base64(audio_16k, sample_rate=16000)
        
        result["processing_latency_ms"] = round(elapsed_ms, 2)
        result["audio_data_url"] = audio_url
        return sanitize_for_json(result)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return sanitize_for_json({
            "status": "error",
            "is_ai_generated": False,
            "verdict": "ANALYSIS_ERROR",
            "verdict_title": "ANALYSIS ERROR OCCURRED",
            "summary": f"Signal processing error: {str(e)}",
            "threat_level": "NORMAL",
            "confidence_pct": 0.0,
            "synthetic_probability": 0.0,
            "telecom_action": "RETRY_RECORDING",
            "biometrics": {}
        })

def add_room_acoustics(sig: np.ndarray, fs: int = 16000, noise_level: float = 0.035, add_reverb: bool = True) -> np.ndarray:
    """Simulates realistic room acoustics: reverberation, desk rumble, and background noise."""
    out = sig.copy()
    if add_reverb:
        delays = [int(fs * 0.022), int(fs * 0.039), int(fs * 0.055)]
        for d, g in zip(delays, [0.32, 0.20, 0.12]):
            if d < len(sig):
                out[d:] += sig[:-d] * g
    t = np.linspace(0, len(sig)/fs, len(sig), endpoint=False)
    rumble = 0.010 * np.sin(2 * np.pi * 50 * t) + 0.006 * np.sin(2 * np.pi * 100 * t)
    noise = np.random.normal(0, noise_level, len(sig))
    combined = out + rumble + noise
    return (combined / (np.max(np.abs(combined)) + 1e-6)).astype(np.float32)

@app.post("/api/analyze-preset")
async def analyze_preset(req: AnalyzePresetRequest):
    """
    Generates and benchmarks realistic Human Voice or AI Voice Clone samples,
    including real-world acoustic room conditions (background noise & reverberation).
    """
    start_t = time.perf_counter()
    fs = 16000
    if req.preset == "human":
        audio = generate_human_voice_sample(duration_sec=2.5, sample_rate=fs)
    elif req.preset == "human_noisy":
        raw = generate_human_voice_sample(duration_sec=2.5, sample_rate=fs)
        audio = add_room_acoustics(raw, fs=fs, noise_level=0.035, add_reverb=True)
    elif req.preset == "ai_noisy":
        raw = generate_synthetic_voice_sample(duration_sec=2.5, sample_rate=fs)
        audio = add_room_acoustics(raw, fs=fs, noise_level=0.035, add_reverb=True)
    else:  # "ai_clone"
        audio = generate_synthetic_voice_sample(duration_sec=2.5, sample_rate=fs)
        
    result = detector.analyze_full_recording(audio, sample_rate=fs)
    elapsed_ms = (time.perf_counter() - start_t) * 1000.0
    audio_url = pcm_to_wav_base64(audio, sample_rate=fs)
    
    result["processing_latency_ms"] = round(elapsed_ms, 2)
    result["audio_data_url"] = audio_url
    result["preset_type"] = req.preset
    return sanitize_for_json(result)

@app.post("/api/process-live-audio")
async def process_live_audio(req: ProcessLiveAudioRequest):
    start_t = time.perf_counter()
    raw_samples = np.array(req.samples, dtype=np.float32)
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
        t = np.linspace(0, 0.4, int(fs * 0.4))
        f0 = 135.0 * (1.0 + 0.018 * np.sin(2 * np.pi * 5.5 * t) + np.random.normal(0, 0.005, len(t)))
        phase = 2 * np.pi * np.cumsum(f0) / fs
        pcm_chunk = np.sin(phase) + 0.5 * np.sin(2 * phase) + 0.25 * np.sin(3 * phase)
    else:
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
    """
    Upload any audio file to test AI detection.
    Supports MP3, WAV, FLAC, OGG, AIFF, and M4A formats via soundfile.
    """
    content = await file.read()
    try:
        audio = None
        sr = 16000
        
        # 1. Primary: High-fidelity universal decoding via soundfile (MP3, WAV, FLAC, OGG, etc.)
        if HAVE_SOUNDFILE:
            try:
                buf = io.BytesIO(content)
                audio_sf, sr = sf.read(buf, dtype='float32')
                if audio_sf.ndim > 1:
                    audio_sf = np.mean(audio_sf, axis=1) # Downmix multi-channel to mono
                audio = audio_sf
            except Exception:
                audio = None

        # 2. Secondary fallback: standard library wave module for uncompressed WAV
        if audio is None:
            try:
                buf = io.BytesIO(content)
                with wave.open(buf, 'rb') as wf:
                    sr = wf.getframerate()
                    n_channels = wf.getnchannels()
                    frames = wf.readframes(wf.getnframes())
                    audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
                    if n_channels > 1:
                        audio = audio[::n_channels]
            except Exception:
                audio = None

        # 3. Last-ditch fallback for raw PCM byte streams
        if audio is None:
            audio = np.frombuffer(content, dtype=np.int16).astype(np.float32) / 32768.0
            sr = 16000

        # Resample to 16kHz if needed
        if sr != 16000 and len(audio) > 10:
            audio = resample_to_16k(audio, sr)

        if len(audio) < 2000:
            return {"status": "error", "error": "Audio file too short (minimum 0.3s required)"}

        result = detector.analyze_full_recording(audio, sample_rate=16000)
        audio_url = pcm_to_wav_base64(audio, sample_rate=16000)
        result["filename"] = file.filename
        result["audio_data_url"] = audio_url
        return sanitize_for_json(result)
    except Exception as e:
        return {"status": "error", "error": str(e)}

if __name__ == "__main__":
    import uvicorn
    print("\n" + "=" * 70)
    print(" TeleGuard AI: Real-Time AI Voice Detection Server")
    print(" Listening at: http://127.0.0.1:8000")
    print("=" * 70 + "\n")
    uvicorn.run(app, host="127.0.0.1", port=8000)
