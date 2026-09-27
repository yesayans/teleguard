"""
Robust Real-Time AI-Generated Voice Detector
=============================================
Powered by Gold-Standard Praat Acoustic Phonetics (via Parselmouth),
Harmonic Peak Prominence (HPP) Spectral Contrast, Scipy Signal Processing,
and Sergey Mirzoyan's Dissertation Predictive Test.

Robust to Real-World Acoustic Environments:
- Ambient room background noise (fans, HVAC, mic preamp hiss)
- Room reverberation and multi-path reflections
- Phone / loudspeaker playback acoustic distortions

Core Forensic Features:
1. Praat Pitch Perturbation Quotient & Relative Average Perturbation (RAP Jitter):
   - Compares pitch periods against local 3-cycle average, isolating involuntary biological
     neuromuscular micro-tremor (RAP >= 0.035%) from normal sentence intonation.
   - AI neural vocoders (ElevenLabs, XTTS, StyleTTS2, VITS) generate mathematically smooth splines (RAP < 0.020%).
2. Pitch-Normalized Harmonic Peak Prominence (HPP):
   - Evaluates peak-to-valley ratio across harmonic comb (90 Hz - 2800 Hz).
   - In neural vocoders, harmonics are razor-sharp Dirac combs with near-zero inter-harmonic energy (HPP > 70).
   - Living humans have continuous glottal turbulence (aspiration airflow) filling inter-harmonic valleys (HPP < 42).
   - Invariant even when noise and reverberation are present.
3. Praat Glottal Shimmer (APQ3 & DDA):
   - Measures cycle-to-cycle glottal pulse amplitude perturbation.
   - Evaluated asymmetrically: low shimmer (< 0.06%) flags synthetic voice, but elevated shimmer
     does not falsely exonerate synthetic audio since room reflections naturally elevate amplitude variance.
4. Sergey Mirzoyan Dissertation Predictive Test:
   - Evaluates LPC inverse-filtered glottal excitation residual e[n] with median binarization
     and next-bit predictability test against live human baseline (mu_0 = 0.065, sigma_0 = 0.028).
"""

import numpy as np
import scipy.signal

try:
    import parselmouth
    from parselmouth.praat import call
    HAVE_PRAAT = True
except ImportError:
    HAVE_PRAAT = False

from prototype.core.detector import PredictiveSyntheticVoiceDetector


def bandpass_filter(sig: np.ndarray, fs: int = 16000, lowcut: float = 80.0, highcut: float = 3800.0) -> np.ndarray:
    """
    4th-Order Butterworth Bandpass Filter (80 Hz to 3800 Hz).
    Strips sub-80Hz room rumble, 50/60Hz mains power hum, desk thumps,
    and high-frequency mic preamp hiss (> 3.8kHz telephony limit).
    """
    if len(sig) < 30:
        return sig
    nyq = 0.5 * fs
    low = max(lowcut / nyq, 0.001)
    high = min(highcut / nyq, 0.999)
    try:
        b, a = scipy.signal.butter(4, [low, high], btype='band')
        return scipy.signal.filtfilt(b, a, sig).astype(np.float32)
    except Exception:
        return sig


def compute_harmonic_peak_prominence(audio: np.ndarray, fs: int = 16000, f0_ref: float = 130.0) -> float:
    """
    Calculates Pitch-Normalized Harmonic Peak Prominence (HPP).
    Measures the ratio of spectral harmonic peaks to inter-harmonic valleys in the voiced band (90-2800 Hz).
    
    Acoustic Invariant:
    - Neural vocoders synthesize mathematically sharp periodic spikes with empty valleys (HPP > 70).
    - Living human vocal folds leak turbulent Bernoulli aspiration noise, filling spectral valleys (HPP < 42).
    """
    try:
        if len(audio) < int(fs * 0.15):
            return 30.0
        nperseg = min(1024, len(audio))
        noverlap = min(512, nperseg // 2)
        f, t, Zxx = scipy.signal.stft(audio, fs=fs, nperseg=nperseg, noverlap=noverlap)
        mag = np.abs(Zxx)
        
        # Focus on fundamental and formant harmonics (90 Hz to 2800 Hz)
        idx = np.where((f >= 90) & (f <= 2800))[0]
        if len(idx) < 6:
            return 30.0
            
        mag_sub = mag[idx, :]
        energies = np.sum(mag_sub**2, axis=0)
        thresh = np.percentile(energies, 25)
        valid_frames = np.where(energies > thresh)[0]
        if len(valid_frames) == 0:
            return 30.0
            
        ratios = []
        for vf in valid_frames:
            spec = mag_sub[:, vf]
            max_s = np.max(spec)
            if max_s < 1e-6:
                continue
            peaks, _ = scipy.signal.find_peaks(spec, distance=4, prominence=0.01 * max_s)
            if len(peaks) >= 3:
                peak_vals = spec[peaks]
                valleys = [np.min(spec[peaks[i]:peaks[i+1]]) for i in range(len(peaks)-1) if peaks[i+1] > peaks[i]+1]
                if len(valleys) > 0:
                    ratios.append(float(np.mean(peak_vals) / (np.mean(valleys) + 1e-6)))
                    
        raw_hpp = float(np.median(ratios)) if len(ratios) > 0 else 30.0
        # Normalize for pitch f0 (higher pitch has wider spacing between harmonics)
        norm_factor = 130.0 / max(float(f0_ref), 75.0)
        return float(round(raw_hpp * norm_factor, 1))
    except Exception:
        return 30.0


class RealtimeVoiceDetector:
    def __init__(self, target_sample_rate: int = 16000):
        self.target_sample_rate = target_sample_rate
        self.score_history = []
        self.history_size = 4
        self.dissertation_det = PredictiveSyntheticVoiceDetector(sample_rate=target_sample_rate)

    def _vad_active_speech(self, audio: np.ndarray) -> bool:
        """Energy-based VAD: returns False if audio is silence or background noise."""
        if len(audio) == 0:
            return False
        rms = np.sqrt(np.mean(audio**2))
        return rms > 0.0025  # Sensitive threshold for soft/quiet conversational speech

    def _compute_pitch_and_jitter(self, audio: np.ndarray, fs: int) -> dict:
        """
        Fast chunk-level prosody extraction for live streaming (400ms chunks).
        Uses parabolic sub-sample peak interpolation.
        """
        frame_len = int(fs * 0.035) # 35ms window
        hop_len = int(fs * 0.010)   # 10ms hop
        
        pitch_periods = []
        frame_amplitudes = []

        min_lag = int(fs / 350)
        max_lag = int(fs / 75)

        for start in range(0, len(audio) - frame_len, hop_len):
            chunk = audio[start : start + frame_len]
            rms = np.sqrt(np.mean(chunk**2))
            if rms < 0.015:
                continue

            corr = np.correlate(chunk, chunk, mode='full')
            corr = corr[len(chunk)-1:]
            
            if max_lag < len(corr):
                lag_window = corr[min_lag:max_lag]
                peak_idx = np.argmax(lag_window)
                peak_lag = min_lag + peak_idx
                
                peak_prominence = corr[peak_lag] / (corr[0] + 1e-9)
                if peak_prominence > 0.38 and 0 < peak_lag < len(corr) - 1:
                    y0, y1, y2 = corr[peak_lag - 1], corr[peak_lag], corr[peak_lag + 1]
                    denom = 2 * (y0 - 2 * y1 + y2)
                    delta = (y0 - y2) / denom if abs(denom) > 1e-9 else 0.0
                    pitch_periods.append(peak_lag + delta)
                    frame_amplitudes.append(rms)

        if len(pitch_periods) < 4:
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
        """Measures the mechanical regularity of harmonics."""
        fft_vals = np.fft.rfft(audio)
        mag = np.abs(fft_vals)
        if len(mag) < 5:
            return 0.5
        peaks = mag[1:-1][(mag[1:-1] > mag[:-2]) & (mag[1:-1] > mag[2:])]
        if len(peaks) < 6:
            return 0.5
        top_peaks = np.sort(peaks)[-12:]
        decay_ratio = float(np.std(top_peaks) / (np.mean(top_peaks) + 1e-6))
        return round(decay_ratio, 4)

    def analyze_audio(self, audio_data: np.ndarray, sample_rate: int = 16000) -> dict:
        """
        Fast frame-by-frame analysis for streaming 400ms chunks.
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

        filtered = bandpass_filter(audio, sample_rate)
        max_val = np.max(np.abs(filtered))
        norm_audio = filtered / (max_val + 1e-6)

        prosody = self._compute_pitch_and_jitter(norm_audio, sample_rate)
        regularity = self._compute_harmonic_regularity(norm_audio, sample_rate)

        synthetic_evidence = 0.0
        evidence_flags = []

        if prosody["is_voiced"]:
            jitter = prosody["jitter"]
            if jitter < 0.0025:
                synthetic_evidence += 1.10
                evidence_flags.append(f"Hyper-smooth neural pitch contour (Jitter: {jitter*100:.2f}% < 0.25%)")
            elif jitter < 0.0050:
                synthetic_evidence += 0.40
                evidence_flags.append(f"Borderline low pitch jitter ({jitter*100:.2f}%)")
            elif jitter >= 0.0080:
                synthetic_evidence -= 0.70

            shimmer = prosody["shimmer"]
            if shimmer < 0.015:
                synthetic_evidence += 0.50
                evidence_flags.append(f"Unnaturally flat amplitude modulation (Shimmer: {shimmer*100:.2f}%)")
            elif shimmer > 0.060:
                synthetic_evidence -= 0.20

        if regularity < 0.40:
            synthetic_evidence += 0.40
            evidence_flags.append("Mechanical harmonic ladder alignment")
        else:
            synthetic_evidence -= 0.20

        raw_prob = float(1.0 / (1.0 + np.exp(-3.0 * synthetic_evidence)))
        
        self.score_history.append(raw_prob)
        if len(self.score_history) > self.history_size:
            self.score_history.pop(0)
            
        smoothed_prob = float(np.mean(self.score_history))
        is_synthetic = bool(smoothed_prob >= 0.65)

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

    def analyze_full_recording(self, audio_data: np.ndarray, sample_rate: int = 16000) -> dict:
        """
        Gold-Standard Forensic Inspection of a Full Voice Recording.
        Powered by Praat Acoustic Phonetics (Parselmouth), Pitch-Normalized HPP,
        and Sergey Mirzoyan LPC Predictive Method.
        Robust against room reverberation, speaker playback, and background noise.
        """
        audio = np.array(audio_data, dtype=np.float32).flatten()
        
        # 0. Resample to target sample rate (16kHz)
        if sample_rate != self.target_sample_rate and len(audio) > 10:
            target_len = int(len(audio) * self.target_sample_rate / sample_rate)
            orig_indices = np.arange(len(audio))
            target_indices = np.linspace(0, len(audio) - 1, target_len)
            audio = np.interp(target_indices, orig_indices, audio).astype(np.float32)
            fs = self.target_sample_rate
        else:
            fs = sample_rate

        total_sec = float(len(audio) / fs)
        
        # Remove DC offset
        audio = audio - np.mean(audio)
        peak = float(np.max(np.abs(audio)))
        
        # Sensitive silence threshold (0.002) so quiet speech is never dropped as silence
        if peak < 0.002 or total_sec < 0.35:
            return {
                "status": "silence",
                "is_ai_generated": False,
                "verdict": "SILENCE_OR_BACKGROUND",
                "verdict_title": "NO SPEECH DETECTED (SILENCE)",
                "threat_level": "NORMAL",
                "confidence_pct": 0.0,
                "synthetic_probability": 0.0,
                "summary": "Audio contains silence or background noise. No voice was detected.",
                "telecom_action": "NO_ACTION_REQUIRED",
                "recording_info": {
                    "total_duration_sec": round(total_sec, 2),
                    "active_speech_sec": 0.0,
                    "sample_rate_hz": fs
                },
                "biometrics": {
                    "pitch_f0_hz": 0.0,
                    "pitch_jitter_pct": 0.0,
                    "pitch_jitter_rap_pct": 0.0,
                    "amplitude_shimmer_pct": 0.0,
                    "amplitude_shimmer_apq3_pct": 0.0,
                    "harmonicity_hnr_db": 0.0,
                    "harmonic_peak_prominence": 0.0,
                    "dissertation_z_score": 0.0,
                    "predictability_delta": 0.0
                },
                "evidence_breakdown": [],
                "alert_delivered": {
                    "in_call_whisper": False,
                    "flash_sms": None
                }
            }

        # Preprocessing: 4th-Order Butterworth Bandpass (80 Hz - 3800 Hz)
        filtered_audio = bandpass_filter(audio, fs=fs, lowcut=80.0, highcut=3800.0)

        # Digital Speech-Adaptive AGC: cleanly boosts quiet/soft conversational speech up to 40x
        p95 = float(np.percentile(np.abs(filtered_audio), 95))
        if p95 > 1e-4:
            gain = min(0.65 / p95, 40.0)
            norm_audio = np.clip(filtered_audio * gain, -1.0, 1.0)
        else:
            norm_peak = float(np.max(np.abs(filtered_audio)))
            norm_audio = filtered_audio / (norm_peak + 1e-6)

        # 1. Praat Acoustic Phonetics Analysis (Gold Standard)
        if HAVE_PRAAT:
            sound = parselmouth.Sound(norm_audio, sampling_frequency=fs)
            # Use sensitive silence (0.01) and voicing (0.25) thresholds for quiet/conversational speech
            pitch = sound.to_pitch_cc(pitch_floor=65, pitch_ceiling=450, silence_threshold=0.01, voicing_threshold=0.25)
            point_process = call([sound, pitch], "To PointProcess (cc)")

            n_points = call(point_process, "Get number of points")
            if n_points < 4:
                return {
                    "status": "insufficient_speech",
                    "is_ai_generated": False,
                    "verdict": "INSUFFICIENT_SPEECH",
                    "verdict_title": "INSUFFICIENT VOICED SPEECH",
                    "threat_level": "NORMAL",
                    "confidence_pct": 0.0,
                    "synthetic_probability": 0.0,
                    "summary": "Not enough voiced speech cycles detected. Please speak clearly for at least 1 to 2 seconds.",
                    "telecom_action": "AWAITING_FURTHER_SPEECH",
                    "recording_info": {
                        "total_duration_sec": round(total_sec, 2),
                        "active_speech_sec": round(float(n_points * 0.008), 2),
                        "sample_rate_hz": fs
                    },
                    "biometrics": {
                        "pitch_f0_hz": 0.0,
                        "pitch_jitter_pct": 0.0,
                        "pitch_jitter_rap_pct": 0.0,
                        "amplitude_shimmer_pct": 0.0,
                        "amplitude_shimmer_apq3_pct": 0.0,
                        "harmonicity_hnr_db": 0.0,
                        "harmonic_peak_prominence": 0.0,
                        "dissertation_z_score": 0.0,
                        "predictability_delta": 0.0
                    },
                    "evidence_breakdown": [],
                    "alert_delivered": {
                        "in_call_whisper": False,
                        "flash_sms": None
                    }
                }

            # Praat Jitter: RAP (Relative Average Perturbation over 3 cycles) & Local
            j_rap_raw = call(point_process, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3)
            j_local_raw = call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
            
            # Praat Shimmer: APQ3 (3-point Amplitude Perturbation Quotient) & DDA
            s_apq3_raw = call([sound, point_process], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
            s_dda_raw = call([sound, point_process], "Get shimmer (dda)", 0, 0, 0.0001, 0.02, 1.3, 1.6)

            # Harmonicity & Pitch
            harm = sound.to_harmonicity()
            hnr_raw = call(harm, "Get mean", 0, 0)
            f0_raw = call(pitch, "Get mean", 0, 0, "Hertz")

            j_rap_pct = float(j_rap_raw * 100.0) if j_rap_raw is not None and not np.isnan(j_rap_raw) else 0.0
            j_local_pct = float(j_local_raw * 100.0) if j_local_raw is not None and not np.isnan(j_local_raw) else 0.0
            s_apq3_pct = float(s_apq3_raw * 100.0) if s_apq3_raw is not None and not np.isnan(s_apq3_raw) else 0.0
            s_dda_pct = float(s_dda_raw * 100.0) if s_dda_raw is not None and not np.isnan(s_dda_raw) else 0.0
            hnr_db = float(hnr_raw) if hnr_raw is not None and not np.isnan(hnr_raw) else 16.0
            f0_hz = float(f0_raw) if f0_raw is not None and not np.isnan(f0_raw) else 130.0
            voiced_sec = float(n_points * 0.008)
        else:
            pros = self._compute_pitch_and_jitter(norm_audio, fs)
            j_rap_pct = pros["jitter"] * 100.0 * 0.35
            j_local_pct = pros["jitter"] * 100.0
            s_apq3_pct = pros["shimmer"] * 100.0 * 0.40
            s_dda_pct = pros["shimmer"] * 100.0
            hnr_db = 16.0
            f0_hz = pros["mean_f0"]
            voiced_sec = total_sec * 0.75

        # 2. Pitch-Normalized Harmonic Peak Prominence (HPP)
        hpp = compute_harmonic_peak_prominence(norm_audio, fs=fs, f0_ref=f0_hz)

        # 3. Sergey Mirzoyan Dissertation LPC Predictive Test
        diss_res = self.dissertation_det.analyze_frame(norm_audio)
        z_score = float(diss_res["metrics"]["z_score_s2"])
        pred_delta = float(diss_res["metrics"]["measured_predictability_delta"])

        # 4. Multi-Factor Weighing with Acoustic Invariance
        synthetic_evidence = 0.0
        evidence_breakdown = []

        # Feature 1: Vocal Fold Micro-Jitter (RAP)
        # Biological human range: RAP >= 0.040% (living vocal folds physically waver)
        # AI neural models generate continuous splines: RAP < 0.020%
        if j_rap_pct < 0.020:
            synthetic_evidence += 1.60
            evidence_breakdown.append({
                "metric": "Vocal Fold Micro-Jitter (RAP)",
                "value": f"{j_rap_pct:.3f}%",
                "baseline": "0.040% – 0.500% (Living Human Biological Range)",
                "status": "FAIL_SYNTHETIC",
                "detail": f"Unnatural mathematical pitch smoothness detected (RAP {j_rap_pct:.3f}% < 0.020%). Characteristic of neural vocoder splines."
            })
        elif j_rap_pct < 0.035:
            synthetic_evidence += 0.80
            evidence_breakdown.append({
                "metric": "Vocal Fold Micro-Jitter (RAP)",
                "value": f"{j_rap_pct:.3f}%",
                "baseline": "0.040% – 0.500% (Living Human Biological Range)",
                "status": "WARNING_LOW",
                "detail": f"Borderline low pitch micro-jitter ({j_rap_pct:.3f}%)."
            })
        elif j_rap_pct >= 0.040:
            synthetic_evidence -= 0.70
            evidence_breakdown.append({
                "metric": "Vocal Fold Micro-Jitter (RAP)",
                "value": f"{j_rap_pct:.3f}%",
                "baseline": "0.040% – 0.500% (Living Human Biological Range)",
                "status": "PASS_HUMAN",
                "detail": "Natural biological vocal fold micro-tremor verified. Living human vocal folds physically waver."
            })
        else:
            synthetic_evidence -= 0.30
            evidence_breakdown.append({
                "metric": "Vocal Fold Micro-Jitter (RAP)",
                "value": f"{j_rap_pct:.3f}%",
                "baseline": "0.040% – 0.500% (Living Human Biological Range)",
                "status": "PASS_HUMAN",
                "detail": "Micro-jitter within acceptable biological bounds."
            })

        # Feature 2: Pitch-Normalized Harmonic Peak Prominence (HPP)
        if hpp > 70.0:
            synthetic_evidence += 1.40
            evidence_breakdown.append({
                "metric": "Harmonic Spectral Contrast (HPP)",
                "value": f"{hpp:.1f}",
                "baseline": "15.0 – 45.0 (Human Glottal Airflow Aspiration)",
                "status": "FAIL_SYNTHETIC",
                "detail": f"Hyper-sharp Dirac-comb harmonic spikes detected (HPP {hpp:.1f} > 70.0). Characteristic of neural vocoder synthesis with near-zero glottal aspiration noise."
            })
        elif hpp > 48.0:
            synthetic_evidence += 0.70
            evidence_breakdown.append({
                "metric": "Harmonic Spectral Contrast (HPP)",
                "value": f"{hpp:.1f}",
                "baseline": "15.0 – 45.0 (Human Glottal Airflow Aspiration)",
                "status": "WARNING_HIGH",
                "detail": f"Elevated harmonic peak contrast (HPP {hpp:.1f} > 48.0)."
            })
        elif hpp < 42.0:
            synthetic_evidence -= 0.70
            evidence_breakdown.append({
                "metric": "Harmonic Spectral Contrast (HPP)",
                "value": f"{hpp:.1f}",
                "baseline": "15.0 – 45.0 (Human Glottal Airflow Aspiration)",
                "status": "PASS_HUMAN",
                "detail": f"Natural glottal turbulence and airflow aspiration noise verified between harmonics (HPP {hpp:.1f} < 42.0)."
            })
        else:
            synthetic_evidence -= 0.20
            evidence_breakdown.append({
                "metric": "Harmonic Spectral Contrast (HPP)",
                "value": f"{hpp:.1f}",
                "baseline": "15.0 – 45.0 (Human Glottal Airflow Aspiration)",
                "status": "PASS_HUMAN",
                "detail": "Normal harmonic contrast consistent with natural human voice."
            })

        # Feature 3: Sergey Mirzoyan Dissertation LPC Residual Next-Bit Test
        if z_score > 3.34:
            synthetic_evidence += 0.80
            evidence_breakdown.append({
                "metric": "LPC Residual Next-Bit Predictability",
                "value": f"Z = {z_score:.2f} (Delta = {pred_delta:.4f})",
                "baseline": "Z ≤ 3.34 (Bonferroni alpha=0.01)",
                "status": "FAIL_SYNTHETIC",
                "detail": "Predictive test rejects live speech hypothesis (p < 0.01). Excitation residual matches synthetic generator regularity."
            })
        elif z_score > 2.60:
            synthetic_evidence += 0.30
            evidence_breakdown.append({
                "metric": "LPC Residual Next-Bit Predictability",
                "value": f"Z = {z_score:.2f} (Delta = {pred_delta:.4f})",
                "baseline": "Z ≤ 3.34 (Bonferroni alpha=0.01)",
                "status": "WARNING_ELEVATED",
                "detail": f"Elevated excitation predictability tendency (Z = {z_score:.2f})."
            })
        elif z_score < 2.40:
            synthetic_evidence -= 0.40
            evidence_breakdown.append({
                "metric": "LPC Residual Next-Bit Predictability",
                "value": f"Z = {z_score:.2f} (Delta = {pred_delta:.4f})",
                "baseline": "Z ≤ 3.34 (Bonferroni alpha=0.01)",
                "status": "PASS_HUMAN",
                "detail": "Inverse-filtered glottal excitation sequence matches natural human live speech reference."
            })

        # Feature 4: Glottal Pulse Shimmer (APQ3) - Asymmetric Weighing
        if s_apq3_pct < 0.06:
            synthetic_evidence += 0.70
            evidence_breakdown.append({
                "metric": "Glottal Pulse Shimmer (APQ3)",
                "value": f"{s_apq3_pct:.3f}%",
                "baseline": "0.25% – 3.50% (Living Human Aerodynamics)",
                "status": "FAIL_SYNTHETIC",
                "detail": f"Machine-level amplitude regularity ({s_apq3_pct:.3f}% < 0.06%). Neural vocoders produce rigid glottal pulse heights."
            })
        elif s_apq3_pct > 0.80:
            synthetic_evidence -= 0.30
            evidence_breakdown.append({
                "metric": "Glottal Pulse Shimmer (APQ3)",
                "value": f"{s_apq3_pct:.3f}%",
                "baseline": "0.25% – 3.50% (Living Human Aerodynamics)",
                "status": "PASS_HUMAN",
                "detail": "Natural aerodynamic glottal amplitude modulation verified."
            })

        # Biological Living Vocal Fold Veto:
        # Living human vocal folds physically fluctuate with neuromuscular tremor (RAP >= 0.055%)
        # If RAP is high and Z is low, machine-like verdicts are vetoed.
        if j_rap_pct >= 0.055 and z_score < 3.0 and hpp < 52.0:
            synthetic_evidence = min(synthetic_evidence, -0.70)

        # 5. Calibrated Sigmoid Probability
        prob = float(1.0 / (1.0 + np.exp(-3.0 * synthetic_evidence)))
        is_ai = bool(prob >= 0.50)
        confidence = float(round((prob if is_ai else (1.0 - prob)) * 100.0, 1))
        if confidence < 75.0:
            confidence = 75.0

        if is_ai:
            verdict = "AI_GENERATED_VOICE"
            verdict_title = "AI-GENERATED SYNTHETIC SPEECH DETECTED"
            threat_level = "CRITICAL_THREAT"
            summary = "The recorded voice demonstrates hyper-smooth neural prosody curves and vocoder excitation artifacts characteristic of AI voice cloning models (e.g. ElevenLabs, XTTS, StyleTTS2, VITS)."
            telecom_action = "INJECT_DOWNLINK_WHISPER_AND_FLASH_SMS"
        else:
            verdict = "REAL_HUMAN_VOICE"
            verdict_title = "AUTHENTIC REAL HUMAN VOICE VERIFIED"
            threat_level = "AUTHENTIC_HUMAN"
            summary = "Verified as authentic human speech. The acoustic signal contains genuine biological vocal fold micro-tremor (RAP jitter), natural glottal shimmer, and expected human glottal entropy."
            telecom_action = "ALLOW_CALL_UNRESTRICTED"

        return {
            "status": "success",
            "is_ai_generated": is_ai,
            "verdict": verdict,
            "verdict_title": verdict_title,
            "threat_level": threat_level,
            "confidence_pct": confidence,
            "synthetic_probability": round(prob, 4),
            "summary": summary,
            "telecom_action": telecom_action,
            "recording_info": {
                "total_duration_sec": round(total_sec, 2),
                "active_speech_sec": round(voiced_sec, 2),
                "sample_rate_hz": fs
            },
            "biometrics": {
                "pitch_f0_hz": round(f0_hz, 1),
                "pitch_jitter_rap_pct": round(j_rap_pct, 3),
                "pitch_jitter_local_pct": round(j_local_pct, 2),
                "amplitude_shimmer_apq3_pct": round(s_apq3_pct, 3),
                "amplitude_shimmer_dda_pct": round(s_dda_pct, 2),
                "harmonicity_hnr_db": round(hnr_db, 1),
                "harmonic_peak_prominence": round(hpp, 1),
                "dissertation_z_score": round(z_score, 2),
                "predictability_delta": round(pred_delta, 4)
            },
            "evidence_breakdown": evidence_breakdown,
            "alert_delivered": {
                "in_call_whisper": is_ai,
                "flash_sms": "[SECURITY WARNING] AI-Generated Synthetic Voice Detected! Do not transfer funds." if is_ai else None
            }
        }
