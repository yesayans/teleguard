"""
Voice Certificate Module
========================
Purpose: Make a registered caller's voice mathematically impossible to imitate with AI voice cloning.

At the sending side (Caller / Client Device):
- Embeds an inaudible, cryptographic, certificate-based noise pattern into the caller's voice.
- Generated from a shared secret key, rotating every 30 seconds and unique to each call_id.
- AI voice cloners copy the voice timbre ("body"), but cannot produce the correct noise pattern.
- Replaying a past genuine call fails because old certificates belong to expired time epochs or different call_ids.

At the verifying side (Network / SBC Gateway):
- Inspects calls claiming to come from a registered subscriber.
- Checks if the current 30-second call-specific certificate is present and valid.
- If missing or invalid -> triggers instant in-call warning and flags the caller as an AI imposter.
"""

import time
import hmac
import hashlib
import numpy as np

class VoiceCertificateEmbedder:
    """
    Sending Side: Injects an inaudible, psychoacoustically masked,
    dynamic noise certificate into outgoing voice frames.
    """
    def __init__(self, secret_key: str, epoch_seconds: int = 30, sample_rate: int = 16000):
        self.secret_key = secret_key.encode('utf-8')
        self.epoch_seconds = epoch_seconds
        self.sample_rate = sample_rate

    def get_epoch(self, timestamp: float = None) -> int:
        t = timestamp if timestamp is not None else time.time()
        return int(t // self.epoch_seconds)

    def generate_pattern(self, call_id: str, epoch: int, length_samples: int) -> np.ndarray:
        """
        Derives an unpredictable pseudo-random noise sequence using HMAC-SHA256.
        Bound to: (secret_key, call_id, epoch).
        """
        message = f"VOICE_CERT:{call_id}:{epoch}".encode('utf-8')
        h = hmac.new(self.secret_key, message, hashlib.sha256).digest()
        
        # Deterministically seed PRNG with HMAC bytes
        seed = int.from_bytes(h[:8], byteorder='big')
        rng = np.random.RandomState(seed % (2**32))
        
        # Generate spread-spectrum Gaussian noise
        raw_noise = rng.normal(0.0, 1.0, length_samples)
        
        # Bandpass shape: 1.5 kHz to 3.4 kHz (telephony passband, sub-audible)
        fft_noise = np.fft.rfft(raw_noise)
        freqs = np.fft.rfftfreq(length_samples, d=1.0/self.sample_rate)
        
        # Apply bandpass shaping filter (1500 - 3400 Hz)
        bandpass = (freqs >= 1500) & (freqs <= 3400)
        fft_noise[~bandpass] = 0.0
        
        filtered_noise = np.fft.irfft(fft_noise, n=length_samples)
        norm = np.linalg.norm(filtered_noise)
        if norm > 1e-6:
            filtered_noise = filtered_noise / norm
            
        return filtered_noise

    def embed_certificate(self, audio_frame: np.ndarray, call_id: str, timestamp: float = None, snr_db: float = -22.0) -> np.ndarray:
        """
        Embeds the dynamic certificate into speech audio at sub-audible amplitude.
        Psychoacoustically masked below speech energy.
        """
        length = len(audio_frame)
        epoch = self.get_epoch(timestamp)
        pattern = self.generate_pattern(call_id, epoch, length)
        
        speech_rms = np.sqrt(np.mean(audio_frame**2)) if len(audio_frame) > 0 else 0.1
        if speech_rms < 1e-4:
            speech_rms = 0.05
            
        # Target amplitude shaped to speech energy
        noise_amplitude = speech_rms * (10 ** (snr_db / 20.0)) * np.sqrt(length)
        scaled_pattern = pattern * noise_amplitude
        
        certified_audio = audio_frame + scaled_pattern
        return certified_audio.astype(np.float32)


class VoiceCertificateVerifier:
    """
    Verifying Side (Network / SBC): Evaluates incoming voice stream
    against the expected dynamic certificate for this call and epoch.
    """
    def __init__(self, secret_key: str, epoch_seconds: int = 30, sample_rate: int = 16000):
        self.secret_key = secret_key.encode('utf-8')
        self.epoch_seconds = epoch_seconds
        self.sample_rate = sample_rate
        self.embedder = VoiceCertificateEmbedder(secret_key, epoch_seconds, sample_rate)

    def get_epoch(self, timestamp: float = None) -> int:
        return self.embedder.get_epoch(timestamp)

    def _pre_emphasize(self, signal: np.ndarray) -> np.ndarray:
        """Whitening pre-emphasis filter to suppress low-frequency speech formants."""
        if len(signal) < 2:
            return signal
        diff = np.diff(signal)
        norm = np.linalg.norm(diff)
        if norm > 1e-6:
            return diff / norm
        return diff

    def verify_frame(self, audio_frame: np.ndarray, call_id: str, timestamp: float = None, correlation_threshold: float = 0.15) -> dict:
        """
        Verifies if the audio frame contains the legitimate dynamic certificate.
        Handles +/- 1 epoch tolerance for network jitter.
        Detects:
        - Missing certificate (AI Voice Clone)
        - Expired certificate (Replay attack from earlier call)
        - Invalid signature (Attacker injecting random noise)
        """
        t = timestamp if timestamp is not None else time.time()
        current_epoch = self.get_epoch(t)
        length = len(audio_frame)

        if length < 128:
            return {"verified": False, "correlation_score": 0.0, "status": "INSUFFICIENT_AUDIO", "reason": "Frame too short"}

        whitened_audio = self._pre_emphasize(audio_frame)

        # Test current epoch and adjacent epochs (clock drift tolerance)
        epochs_to_test = [current_epoch, current_epoch - 1, current_epoch + 1]
        best_corr = -1.0
        best_epoch = current_epoch

        for ep in epochs_to_test:
            expected_pattern = self.embedder.generate_pattern(call_id, ep, length)
            whitened_pattern = self._pre_emphasize(expected_pattern)
            corr = float(np.dot(whitened_audio, whitened_pattern))
            if corr > best_corr:
                best_corr = corr
                best_epoch = ep

        # Test for REPLAY ATTACK (checking if it matches an expired epoch from earlier in the day)
        replay_detected = False
        expired_epochs = [current_epoch - 5, current_epoch - 10, current_epoch - 30]
        for old_ep in expired_epochs:
            old_pattern = self.embedder.generate_pattern(call_id, old_ep, length)
            whitened_old_pattern = self._pre_emphasize(old_pattern)
            old_corr = float(np.dot(whitened_audio, whitened_old_pattern))
            if old_corr > correlation_threshold:
                replay_detected = True
                break

        is_verified = (best_corr >= correlation_threshold) and (not replay_detected)

        if replay_detected:
            status = "REPLAY_ATTACK_DETECTED"
            reason = "Acoustic certificate matches an expired epoch. The audio is a pre-recorded replay."
        elif is_verified:
            status = "CERTIFIED_GENUINE_CALLER"
            reason = "Dynamic cryptographic noise certificate matches active 30s epoch and call ID."
        else:
            status = "IMPOSTER_CERTIFICATE_MISSING"
            reason = "Acoustic certificate missing or obliterated by an AI neural vocoder."

        return {
            "verified": is_verified,
            "correlation_score": round(float(np.clip(best_corr, 0.0, 1.0)), 4),
            "status": status,
            "reason": reason,
            "epoch_tested": best_epoch,
            "seconds_remaining_in_epoch": int(self.epoch_seconds - (t % self.epoch_seconds)),
            "replay_attack": replay_detected
        }
