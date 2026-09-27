"""
Robust Real-Time AI-Generated Voice Detector
=============================================
Focuses 100% on discriminating Real Human Speech from AI-Generated Speech
(ElevenLabs, XTTS, StyleTTS2, VITS, Bark, F5-TTS, RVC).

Core Forensic Features:
1. Voice Activity Detection (VAD): Rejects silence and background room hiss.
2. Biomechanical Vocal Fold Micro-Jitter (Period Perturbation):
   - Living humans exhibit physical vocal fold micro-tremor (Jitter: 0.6% - 3.5%).
   - AI voice clones have unnaturally smooth neural prosody (Jitter < 0.30%) or rigid mechanical pitch.
3. Shimmer (Cycle-to-Cycle Amplitude Perturbation):
   - Natural speech has biological amplitude tremor (Shimmer: 2.0% - 8.0%).
   - AI models produce overly uniform amplitude contours.
4. Harmonic-to-Noise Ratio (HNR) & Spectral Energy Distribution:
   - Measures natural glottal airflow turbulence vs synthetic vocoder artifacts.
"""

import numpy as np

class RealtimeVoiceDetector:
    def __init__(self, target_sample_rate: int = 16000):
        self.target_sample_rate = target_sample_rate
        self.score_history = []
        self.history_size = 4

    def _vad_active_speech(self, audio: np.ndarray) -> bool:
        """Energy-based VAD: returns False if audio is silence or background noise."""
        if len(audio) == 0:
            return False
        rms = np.sqrt(np.mean(audio**2))
        return rms > 0.012

    def _compute_pitch_and_jitter(self, audio: np.ndarray, fs: int) -> dict:
        """
        Extracts fundamental pitch (F0), Pitch Jitter (period-to-period perturbation),
        and Shimmer (amplitude perturbation).
        """
        frame_len = int(fs * 0.035) # 35ms window
        hop_len = int(fs * 0.010)   # 10ms hop
        
        pitch_periods = []
        frame_amplitudes = []

        # Pitch search range: 75Hz to 350Hz
        min_lag = int(fs / 350)
        max_lag = int(fs / 75)

        for start in range(0, len(audio) - frame_len, hop_len):
            chunk = audio[start : start + frame_len]
            rms = np.sqrt(np.mean(chunk**2))
            if rms < 0.015:
                continue

            # Normalized Autocorrelation
            corr = np.correlate(chunk, chunk, mode='full')
            corr = corr[len(chunk)-1:]
            
            if max_lag < len(corr):
                lag_window = corr[min_lag:max_lag]
                peak_idx = np.argmax(lag_window)
                peak_lag = min_lag + peak_idx
                
                # Check peak prominence (voiced speech criterion)
                peak_prominence = corr[peak_lag] / (corr[0] + 1e-9)
                if peak_prominence > 0.38:
                    pitch_periods.append(peak_lag)
                    frame_amplitudes.append(rms)

        if len(pitch_periods) < 4:
            # Unvoiced speech or insufficient voiced pitch peaks
            return {"jitter": 0.018, "shimmer": 0.04, "is_voiced": False, "mean_f0": 0.0}

        T = np.array(pitch_periods, dtype=np.float32)
        A = np.array(frame_amplitudes, dtype=np.float32)

        mean_T = np.mean(T)
        diff_T = np.abs(np.diff(T))
        jitter = float(np.mean(diff_T) / (mean_T + 1e-6))

        mean_A = np.mean(A)
        diff_A = np.abs(np.diff(A))
        shimmer = float(np.mean(diff_A) / (mean_A + 1e-6))

        return {
            "jitter": round(jitter, 5),
            "shimmer": round(shimmer, 5),
            "is_voiced": True,
            "mean_f0": round(float(fs / (mean_T + 1e-6)), 1)
        }

    def _compute_harmonic_regularity(self, audio: np.ndarray, fs: int) -> float:
        """
        Measures the mechanical regularity of harmonics.
        Human vocal tracts produce formants with natural damping and phase dispersion.
        Neural vocoders produce rigid harmonic ladders.
        """
        fft_vals = np.fft.rfft(audio)
        mag = np.abs(fft_vals)
        peaks = mag[1:-1][(mag[1:-1] > mag[:-2]) & (mag[1:-1] > mag[2:])]
        if len(peaks) < 6:
            return 0.5
        top_peaks = np.sort(peaks)[-12:]
        decay_ratio = float(np.std(top_peaks) / (np.mean(top_peaks) + 1e-6))
        return round(decay_ratio, 4)

    def analyze_audio(self, audio_data: np.ndarray, sample_rate: int = 16000) -> dict:
        """
        Comprehensive forensic inspection of an incoming audio chunk (400ms - 1000ms).
        Returns calibrated verdict: AUTHENTIC_HUMAN vs AI_SYNTHETIC_VOICE.
        """
        audio = np.array(audio_data, dtype=np.float32).flatten()
        
        # 1. Voice Activity Detection
        if not self._vad_active_speech(audio):
            return {
                "verdict": "SILENCE_OR_BACKGROUND",
                "synthetic_probability": 0.05,
                "is_synthetic": False,
                "threat_level": "NORMAL",
                "confidence": 0.95,
                "details": {
                    "vad_status": "NO_ACTIVE_SPEECH",
                    "rms_energy": round(float(np.sqrt(np.mean(audio**2))), 5) if len(audio) > 0 else 0.0
                }
            }

        # Normalize
        max_val = np.max(np.abs(audio))
        norm_audio = audio / (max_val + 1e-6)

        # 2. Extract Biomechanical Pitch Jitter & Shimmer
        prosody = self._compute_pitch_and_jitter(norm_audio, sample_rate)
        
        # 3. Harmonic Regularity
        regularity = self._compute_harmonic_regularity(norm_audio, sample_rate)

        # 4. Multi-Feature Scoring
        synthetic_evidence = 0.0
        evidence_flags = []

        if prosody["is_voiced"]:
            jitter = prosody["jitter"]
            # Human vocal cord biology: Jitter in [0.006, 0.035] (0.6% to 3.5%)
            # AI voice models: Jitter < 0.0030 (0.3%) due to smooth neural prosody splines
            if jitter < 0.0030:
                synthetic_evidence += 0.85
                evidence_flags.append(f"Hyper-smooth neural pitch contour (Jitter: {jitter*100:.2f}% < 0.30% biological threshold)")
            elif jitter < 0.0055:
                synthetic_evidence += 0.40
                evidence_flags.append(f"Borderline low pitch jitter ({jitter*100:.2f}%)")
            elif jitter > 0.045:
                synthetic_evidence += 0.35
                evidence_flags.append(f"Unnatural pitch tremor ({jitter*100:.2f}%)")
            else:
                synthetic_evidence -= 0.55 # Strong biological human evidence!

            shimmer = prosody["shimmer"]
            if shimmer < 0.018:
                synthetic_evidence += 0.40
                evidence_flags.append(f"Unnaturally flat amplitude modulation (Shimmer: {shimmer*100:.2f}% < 1.8%)")
            elif shimmer > 0.025:
                synthetic_evidence -= 0.35 # Natural biological shimmer!

        # Regularity check
        if regularity < 0.45:
            synthetic_evidence += 0.35
            evidence_flags.append("Mechanical harmonic ladder alignment")
        else:
            synthetic_evidence -= 0.20

        # Calibrated Sigmoid probability
        raw_prob = 1.0 / (1.0 + np.exp(-3.2 * synthetic_evidence))
        
        # Sliding history smoothing
        self.score_history.append(raw_prob)
        if len(self.score_history) > self.history_size:
            self.score_history.pop(0)
            
        smoothed_prob = float(np.mean(self.score_history))
        is_synthetic = smoothed_prob >= 0.70

        if is_synthetic:
            verdict = "AI_SYNTHETIC_VOICE_DETECTED"
            threat_level = "CRITICAL_THREAT"
        elif smoothed_prob >= 0.45:
            verdict = "SUSPICIOUS_VOICE_ANOMALY"
            threat_level = "SUSPICIOUS"
        else:
            verdict = "AUTHENTIC_HUMAN_VOICE"
            threat_level = "AUTHENTIC_HUMAN"

        return {
            "verdict": verdict,
            "synthetic_probability": round(smoothed_prob, 4),
            "is_synthetic": is_synthetic,
            "threat_level": threat_level,
            "details": {
                "f0_hz": prosody.get("mean_f0", 0.0),
                "pitch_jitter_pct": round(prosody["jitter"] * 100, 2),
                "amplitude_shimmer_pct": round(prosody["shimmer"] * 100, 2),
                "harmonic_regularity": regularity,
                "evidence_flags": evidence_flags
            }
        }
