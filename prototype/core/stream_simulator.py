"""
Telecom In-Network Streaming Pipeline Simulator
Implements:
1. EphemeralCircularRingBuffer: In-memory sliding buffer (zero audio recording on disk, GDPR compliant).
2. Real-Time Multi-Tier Inspection Engine.
3. In-Call Whisper & Flash-SMS Alert Generator.
"""

import time
import numpy as np
from prototype.core.watermark import WatermarkDetector
from prototype.core.detector import AcousticReconstructionDetector
from prototype.core.certificate import SubscriberAcousticCertificate

from prototype.core.voice_certificate import VoiceCertificateVerifier, VoiceCertificateEmbedder

class EphemeralCircularRingBuffer:
    """
    Fixed-size volatile RAM ring buffer.
    Guarantees that audio data is never written to persistent disk storage.
    Old audio packets are immediately overwritten as the head pointer advances.
    """
    def __init__(self, capacity_samples: int = 16000 * 2): # 2 seconds of 16kHz audio
        self.capacity = capacity_samples
        self.buffer = np.zeros(capacity_samples, dtype=np.float32)
        self.head = 0
        self.total_samples_written = 0

    def push(self, pcm_chunk: np.ndarray):
        chunk_len = len(pcm_chunk)
        if chunk_len >= self.capacity:
            self.buffer[:] = pcm_chunk[-self.capacity:]
            self.head = 0
        else:
            end_pos = (self.head + chunk_len) % self.capacity
            if self.head + chunk_len <= self.capacity:
                self.buffer[self.head : self.head + chunk_len] = pcm_chunk
            else:
                part1 = self.capacity - self.head
                self.buffer[self.head : self.capacity] = pcm_chunk[:part1]
                self.buffer[: chunk_len - part1] = pcm_chunk[part1:]
            self.head = end_pos
        self.total_samples_written += chunk_len

    def get_latest_window(self, num_samples: int) -> np.ndarray:
        """Returns the most recent N samples from volatile RAM without copying to disk."""
        num_samples = min(num_samples, self.capacity)
        start_pos = (self.head - num_samples) % self.capacity
        if start_pos + num_samples <= self.capacity:
            return self.buffer[start_pos : start_pos + num_samples].copy()
        else:
            part1 = self.capacity - start_pos
            part2 = num_samples - part1
            return np.concatenate((self.buffer[start_pos:], self.buffer[:part2]))


class TelecomCallPipeline:
    def __init__(self, subscriber_id: str = "+37491001122", registered_key: str = "GRANDSON_REGISTERED_SECRET_99"):
        self.subscriber_id = subscriber_id
        self.registered_key = registered_key
        self.ring_buffer = EphemeralCircularRingBuffer(capacity_samples=32000)
        self.watermark_detector = WatermarkDetector()
        self.reconstruction_detector = AcousticReconstructionDetector(sample_rate=16000)
        self.certificate_engine = SubscriberAcousticCertificate(subscriber_id=subscriber_id)
        
        # New Voice Certificate Verifier & Embedder
        self.voice_cert_verifier = VoiceCertificateVerifier(secret_key=registered_key, epoch_seconds=30)
        self.voice_cert_embedder = VoiceCertificateEmbedder(secret_key=registered_key, epoch_seconds=30)
        
        self.call_state = {
            "frames_processed": 0,
            "threat_level": "NORMAL", # NORMAL, SUSPICIOUS, CRITICAL
            "synthetic_score": 0.0,
            "alert_triggered": False,
            "alert_type": None,
            "active_challenge": None
        }

    def process_incoming_rtp_chunk(self, pcm_data: np.ndarray, response_latency_ms: float = 250.0,
                                   is_registered_caller_claim: bool = False, call_id: str = "CALL_SESSION_DEFAULT") -> dict:
        """
        Processes a live 400ms RTP voice chunk from the telecom switch.
        Runs asynchronously as a side-channel tap (0 delay added to call).
        If the caller claims to be a registered subscriber, Voice Certificate verification takes precedence.
        """
        start_time = time.perf_counter()
        self.ring_buffer.push(pcm_data)
        window = self.ring_buffer.get_latest_window(6400) # 400ms @ 16kHz
        self.call_state["frames_processed"] += 1

        voice_cert_result = None

        # Check Voice Certificate if the inbound caller claims a registered identity
        if is_registered_caller_claim:
            voice_cert_result = self.voice_cert_verifier.verify_frame(window, call_id=call_id)
            detection_tier = "TIER_VOICE_CERTIFICATE"
            
            if voice_cert_result["verified"]:
                synthetic_score = 0.04
                threat_level = "AUTHENTIC_CERTIFIED_CALLER"
                alert_triggered = False
                stage_details = voice_cert_result
            elif voice_cert_result["replay_attack"]:
                synthetic_score = 1.00
                threat_level = "REPLAY_ATTACK_DETECTED"
                alert_triggered = True
                stage_details = voice_cert_result
            else:
                synthetic_score = 0.99
                threat_level = "AI_IMPOSTER_CERTIFICATE_MISSING"
                alert_triggered = True
                stage_details = voice_cert_result
        else:
            # Tier 1: Check Commercial Watermarks (SynthID, AudioSeal, ElevenLabs)
            wm_res = self.watermark_detector.scan(window)
            
            detection_tier = "TIER_1_WATERMARK" if wm_res["detected"] else "TIER_2_ACOUSTIC_RECONSTRUCTION"
            
            if wm_res["detected"]:
                synthetic_score = wm_res["confidence"]
                stage_details = {"provider": wm_res["provider"], "watermark_present": True}
            else:
                # Tier 2: Check Generative Reconstructibility & Vocoder Regularities (Dissertation Method)
                dsp_res = self.reconstruction_detector.analyze_frame(window)
                synthetic_score = dsp_res["synthetic_probability"]
                stage_details = dsp_res["metrics"]

            # Threat Level determination
            if synthetic_score >= 0.80:
                threat_level = "CRITICAL_THREAT"
                alert_triggered = True
            elif synthetic_score >= 0.55:
                threat_level = "SUSPICIOUS"
                alert_triggered = False
            else:
                threat_level = "AUTHENTIC_HUMAN"
                alert_triggered = False

        alert_type = "IN_CALL_AUDIO_WHISPER_AND_FLASH_SMS" if alert_triggered else None

        self.call_state["threat_level"] = threat_level
        self.call_state["synthetic_score"] = round(float(synthetic_score), 4)
        self.call_state["alert_triggered"] = alert_triggered
        self.call_state["alert_type"] = alert_type

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        return {
            "frame_index": self.call_state["frames_processed"],
            "processing_latency_ms": round(elapsed_ms, 2),
            "threat_level": threat_level,
            "synthetic_score": self.call_state["synthetic_score"],
            "detection_tier_used": detection_tier,
            "stage_details": stage_details,
            "voice_certificate_result": voice_cert_result,
            "alert_delivered": {
                "in_call_whisper": alert_triggered,
                "flash_sms": f"[SECURITY ALERT] TELECOM SHIELD: {threat_level}. Hang up immediately." if alert_triggered else None
            },
            "zero_recording_guarantee": "EPHEMERAL_RAM_ONLY_NO_DISK_IO"
        }
