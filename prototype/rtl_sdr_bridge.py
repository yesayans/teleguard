"""
RTL-SDR Audio Bridge for TeleGuard AI
Captures over-the-air voice signals using an RTL-SDR v2 USB dongle,
demodulates the RF audio, and feeds it into the TeleGuard AI detection pipeline.
"""

import os
import sys
import time
import numpy as np

# Ensure root workspace is on python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from prototype.core.stream_simulator import TelecomCallPipeline

class RTLSDRVoiceBridge:
    def __init__(self, center_freq_hz: float = 433.92e6, sample_rate: int = 2.048e6):
        self.center_freq_hz = center_freq_hz
        self.sample_rate = sample_rate
        self.pipeline = TelecomCallPipeline(subscriber_id="RF_CHANNEL_433MHZ")
        self.sdr_device = None
        self._init_hardware()

    def _init_hardware(self):
        """Attempts to initialize physical RTL-SDR hardware via pyrtlsdr."""
        try:
            from rtlsdr import RtlSdr
            self.sdr_device = RtlSdr()
            self.sdr_device.sample_rate = self.sample_rate
            self.sdr_device.center_freq = self.center_freq_hz
            self.sdr_device.gain = 'auto'
            print(f"[RTL-SDR HARDWARE INITIALIZED] Tuned to: {self.center_freq_hz / 1e6:.2f} MHz")
        except Exception as e:
            print(f"[RTL-SDR NOTICE] Hardware or 'pyrtlsdr' not detected ({e}).")
            print("[RTL-SDR MODE] Running in RF Channel Simulation mode (modeling air-interface fading & noise).")
            self.sdr_device = None

    def capture_and_demodulate_frame(self, duration_sec: float = 0.4) -> np.ndarray:
        """
        Captures RF samples and performs FM demodulation to extract baseband audio.
        If no physical dongle is connected, simulates an RF-transmitted audio channel.
        """
        target_audio_samples = int(16000 * duration_sec)
        
        if self.sdr_device is not None:
            try:
                # Read raw IQ samples from RTL-SDR
                num_iq_samples = int(self.sample_rate * duration_sec)
                iq_samples = self.sdr_device.read_samples(num_iq_samples)
                
                # FM Demodulation: phase difference between successive complex samples
                angle_diff = np.angle(iq_samples[1:] * np.conj(iq_samples[:-1]))
                
                # Decimate / downsample to 16kHz audio
                decimation = int(len(angle_diff) / target_audio_samples)
                audio_demod = angle_diff[::decimation][:target_audio_samples]
                
                # Normalize audio
                if np.max(np.abs(audio_demod)) > 1e-6:
                    audio_demod /= np.max(np.abs(audio_demod))
                return audio_demod.astype(np.float32)
            except Exception as err:
                print(f"[RTL-SDR ERROR] Failed reading hardware: {err}")
                self.sdr_device = None

        # Fallback / Simulated RF Channel:
        # Generates voice with radio channel fading, Rayleigh multipath, and receiver thermal noise
        t = np.linspace(0, duration_sec, target_audio_samples, endpoint=False)
        base_signal = np.sin(2 * np.pi * 140 * t) + 0.5 * np.sin(2 * np.pi * 280 * t)
        
        # RF Rayleigh fading envelope
        rf_fading = 0.8 + 0.2 * np.sin(2 * np.pi * 2.5 * t)
        # Receiver thermal noise floor (SNR ~ 22dB)
        rf_noise = np.random.normal(0, 0.05, len(t))
        
        simulated_rf_audio = (base_signal * rf_fading) + rf_noise
        simulated_rf_audio /= np.max(np.abs(simulated_rf_audio))
        return simulated_rf_audio.astype(np.float32)

    def run_live_rf_listener(self, max_frames: int = 10):
        print("=" * 72)
        print(" TELEGUARD AI: RTL-SDR OVER-THE-AIR VOICE MONITOR")
        print(f" Listening on Frequency: {self.center_freq_hz / 1e6:.2f} MHz (Demodulating 16kHz Audio)")
        print("=" * 72)

        for i in range(1, max_frames + 1):
            audio_frame = self.capture_and_demodulate_frame(duration_sec=0.4)
            result = self.pipeline.process_incoming_rtp_chunk(audio_frame, response_latency_ms=210.0)
            
            print(f" RF Frame {i:02d} | Latency: {result['processing_latency_ms']:5.2f}ms | "
                  f"Threat: {result['threat_level']:<16} | Synthetic Score: {result['synthetic_score']:.4f} | "
                  f"Whisper Alert: {'[ACTIVE]' if result['alert_delivered']['in_call_whisper'] else '[OFF]'}")
            time.sleep(0.3)

        print("\n[RTL-SDR DEMO COMPLETE] Over-the-air audio processed with zero recording.")

if __name__ == "__main__":
    bridge = RTLSDRVoiceBridge(center_freq_hz=433.92e6)
    bridge.run_live_rf_listener(max_frames=6)
