"""
Stage 2: Predictive Test for Synthetic Voice (Sergey Mirzoyan's Dissertation Method)
Reference: "Predictive test for synthetic voice" (Sep 27, 2026 - @Sergey)

Implements:
1. Source-Filter Model & LPC Residual:
   e[n] = x[n] - sum_{k=1}^p a_k x[n-k]  approx G * u[n]
2. Frame-Median Binarization:
   b[n] = 1 if e[n] > median(e) else 0  (P(b=1) = 0.5 by construction)
3. Bit-Level Next-Bit Predictor:
   f_theta(b[n-w], ..., b[n-1]) -> b[n],  delta = p_hat - 0.5
4. Variant B: Pitch Period Jitter Analysis (AMR-safe bitstream metric):
   d_i = (T_{i+1} - T_i) / T_bar,  rho = explained variance ratio
5. Two-Sided Statistical Test against Live-Speech Reference (mu_0, sigma_0):
   Z_j = (delta_j - mu_0) / sigma_0,  s_2 = max |Z_j| > z_{alpha / (2m)}
"""

import numpy as np

class PredictiveSyntheticVoiceDetector:
    def __init__(self, sample_rate: int = 16000, p_lpc: int = 16, window_size: int = 16):
        self.sample_rate = sample_rate
        # p = 10 at 8 kHz (AMR-NB), p = 16 at 16 kHz (AMR-WB / EVS)
        self.p_lpc = p_lpc if sample_rate >= 16000 else 10
        self.window_size = window_size
        
        # Empirical live human baseline reference (calibrated on live speech in telephone channels)
        # Live human residual carries natural vocal fold periodicity
        self.mu_0 = 0.0650
        self.sigma_0 = 0.0280
        
        # Bonferroni-corrected threshold at alpha = 0.01 for m=12 checks (quantile = 3.34)
        self.z_threshold = 3.34

    def compute_lpc_residual(self, audio: np.ndarray) -> np.ndarray:
        p = self.p_lpc
        if len(audio) <= p * 2:
            return np.zeros_like(audio)

        r = np.correlate(audio, audio, mode='full')
        r = r[len(audio) - 1 : len(audio) - 1 + p + 1]
        
        if r[0] < 1e-9:
            return audio

        a = np.zeros(p + 1)
        a[0] = 1.0
        e = r[0]
        for i in range(1, p + 1):
            k = -np.dot(a[:i], r[i:0:-1]) / e
            a[1:i+1] = a[1:i+1] + k * a[i-1::-1]
            a[i] = k
            e *= (1.0 - k**2)
            if e <= 0:
                break

        residual = np.convolve(audio, a, mode='valid')
        return residual

    def binarize_residual(self, residual: np.ndarray) -> np.ndarray:
        if len(residual) == 0:
            return np.array([], dtype=np.int32)
        med = float(np.median(residual))
        return (residual > med).astype(np.int32)

    def next_bit_predictability_test(self, bits: np.ndarray) -> float:
        w = self.window_size
        n = len(bits)
        if n <= w + 50:
            return 0.0

        split = int(0.80 * (n - w))
        X = np.array([bits[i : i + w] for i in range(n - w)])
        y = bits[w : n]

        X_train, y_train = X[:split], y[:split]
        X_test, y_test = X[split:], y[split:]

        # Linear decision boundary estimation on bit windows (MLP analogue)
        weights = np.mean(X_train * (2 * y_train[:, None] - 1), axis=0)
        logits = np.dot(X_test, weights)
        preds = (logits > 0).astype(np.int32)
        
        p_hat = float(np.mean(preds == y_test))
        delta = p_hat - 0.5
        return delta

    def compute_variant_b_pitch_jitter(self, audio: np.ndarray) -> dict:
        frame_len = int(self.sample_rate * 0.025) # 25ms
        hop_len = int(self.sample_rate * 0.005)   # 5ms (AMR subframe)
        
        pitch_lags = []
        for start in range(0, len(audio) - frame_len, hop_len):
            frame = audio[start : start + frame_len]
            if np.std(frame) < 1e-4:
                continue
            corr = np.correlate(frame, frame, mode='full')
            corr = corr[len(frame)-1:]
            
            min_lag = int(self.sample_rate / 350)
            max_lag = int(self.sample_rate / 75)
            if max_lag < len(corr):
                lag = min_lag + np.argmax(corr[min_lag:max_lag])
                pitch_lags.append(float(lag))

        if len(pitch_lags) < 6:
            return {"jitter_ratio_rho": 0.008, "is_synthetic": False}

        T = np.array(pitch_lags)
        T_bar = np.mean(T)
        if T_bar < 1e-4:
            return {"jitter_ratio_rho": 0.008, "is_synthetic": False}

        d = np.diff(T) / T_bar
        var_d = float(np.var(d))
        
        # In live humans, vocal folds exhibit biological micro-jitter: var(d) in [0.0015, 0.035]
        # In neural vocoders, prosody models produce unnaturally smooth pitch (< 0.0008) or robotic jumps
        is_synthetic = (var_d < 0.0009) or (var_d > 0.05)

        return {
            "jitter_ratio_rho": round(var_d, 6),
            "is_synthetic": is_synthetic
        }

    def analyze_frame(self, audio_frame: np.ndarray) -> dict:
        residual = self.compute_lpc_residual(audio_frame)
        bits = self.binarize_residual(residual)
        delta_measured = self.next_bit_predictability_test(bits)
        
        # Z-statistic against human reference
        z_stat = (delta_measured - self.mu_0) / self.sigma_0
        s2 = abs(z_stat)
        
        variant_b = self.compute_variant_b_pitch_jitter(audio_frame)
        
        # Two-sided Bonferroni hypothesis test from dissertation:
        # H0: voice is live (s2 <= z_threshold)
        # H1: voice is synthetic (s2 > z_threshold = 3.34 at alpha = 0.01)
        is_suspicious_a = s2 > self.z_threshold
        is_suspicious_b = variant_b["is_synthetic"] and (s2 > 2.0)

        if is_suspicious_a:
            # Significant deviation from live human reference (more or less predictable)
            synth_score = 0.85 + 0.14 * (1.0 / (1.0 + np.exp(-1.5 * (s2 - self.z_threshold))))
        elif is_suspicious_b:
            synth_score = 0.75
        else:
            # Within normal live human predictability range
            synth_score = 0.15 + 0.25 * (s2 / self.z_threshold)

        return {
            "synthetic_probability": round(float(np.clip(synth_score, 0.05, 0.99)), 4),
            "is_suspicious": synth_score > 0.65,
            "metrics": {
                "measured_predictability_delta": round(delta_measured, 4),
                "human_reference_mu0": self.mu_0,
                "z_score_s2": round(s2, 2),
                "dissertation_threshold_quantile": self.z_threshold,
                "pitch_jitter_variance": variant_b["jitter_ratio_rho"],
                "variant_a_verdict": "SYNTHETIC_ANOMALY" if is_suspicious_a else "LIVE_HUMAN_RANGE",
                "variant_b_verdict": "SYNTHETIC_PITCH_ANOMALY" if variant_b["is_synthetic"] else "NATURAL_BIOLOGICAL_JITTER"
            }
        }

AcousticReconstructionDetector = PredictiveSyntheticVoiceDetector
