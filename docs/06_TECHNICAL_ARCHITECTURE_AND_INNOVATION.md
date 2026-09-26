# Milestone 3: Technical Architecture & Core Innovation

## 1. High-Level System Architecture

The solution operates as an **In-Network Passive Side-Channel Inspection Engine** integrated at the mobile operator's core Session Border Controller (SBC) and IP Multimedia Subsystem (IMS) layer.

```
                           TELECOM CORE NETWORK (IMS / VoLTE / 5G)
+-----------------------------------------------------------------------------------------------+
|                                                                                               |
|  [CALLER] ----(RTP Stream A)----> [SBC / Media Gateway] ----(RTP Stream A)----> [SUBSCRIBER]   |
|                                         |                                            ^        |
|                                         | (Bifurcated Mirror Stream - Zero Latency)  |        |
|                                         v                                            |        |
|                       +-----------------------------------+                          |        |
|                       |  TELEGUARD IN-NETWORK ENGINE      |                          |        |
|                       +-----------------------------------+                          |        |
|                                         |                                            |        |
|               +-------------------------v-------------------------+                  |        |
|               | 1. Ephemeral Volatile RAM Ring Buffer              |                  |        |
|               |    - 20ms Frame Ingestion / 400ms Sliding Window  |                  |        |
|               |    - Zero Disk Storage; GDPR Compliant            |                  |        |
|               +-------------------------+-------------------------+                  |        |
|                                         |                                            |        |
|               +-------------------------v-------------------------+                  |        |
|               | 2. Stage 1: Watermark Decoder                     |                  |        |
|               |    - Meta AudioSeal, SynthID, ElevenLabs Markers  |                  |        |
|               +-------------------------+-------------------------+                  |        |
|                                         | (If Watermark Absent)                      |        |
|               +-------------------------v-------------------------+                  |        |
|               | 3. Stage 2: Generative Reconstructibility & DSP   |                  |        |
|               |    - Phase Spectrum Discontinuity (Vocoder)       |                  |        |
|               |    - Bispectral Bicoherence & Glottal LPC Residual|                  |        |
|               |    - Machine Regularity & Inversion Error         |                  |        |
|               +-------------------------+-------------------------+                  |        |
|                                         | (If Suspicion Score > 0.75)                |        |
|               +-------------------------v-------------------------+                  |        |
|               | 4. Stage 3: Noise-Based Acoustic Certificate      |                  |        |
|               |    - Dynamic Acoustic Challenge-Response          |                  |        |
|               |    - Physical Acoustic Transfer Function Check    |                  |        |
|               +-------------------------+-------------------------+                  |        |
|                                         |                                            |        |
|                               (Threat Detected: Score > 0.85)                        |        |
|                                         v                                            |        |
|               +---------------------------------------------------+                  |        |
|               | 5. Zero-App In-Call Intervention Subsystem        |                  |        |
|               |    - RTP Audio Injection: In-Call Whisper Chime --+------------------+        |
|               |    - Core Signaling: Class-0 Flash SMS / USSD Notification                    |
|               +-------------------------------------------------------------------------------+
```

---

## 2. In-Network Integration Without Delaying Audio

A common misconception is that security inspection must sit *inline* and delay voice packets. TeleGuard uses **Bifurcated RTP Stream Tapping**:
*   The primary audio stream ($RTP_A$) travels from Caller to Callee with zero added delay (<2ms jitter buffer overhead).
*   The Session Border Controller (e.g., Kamailio, Ribbon, AudioCodes, or Asterisk/FreeSWITCH) clones incoming RTP packets into a local high-performance Unix domain socket.
*   The TeleGuard daemon consumes audio in **400ms sliding windows**.
*   **Latency Budget:** Total inference time per window is **< 45 milliseconds** on edge GPU or vectorized CPU (AVX-512 / TensorRT). Within the first **1.2 to 2.0 seconds** of speech, a synthetic voice is flagged and intervention is triggered before the scammer can establish psychological control.

---

## 3. Privacy-by-Design: The Ephemeral Zero-Recording Guarantee

*   **Legal Compliance:** Wiretapping laws (Title III in the US, ePrivacy Directive 2002/58/EC and GDPR Art. 5(1)(c) in the EU) strictly prohibit recording telephone calls without consent.
*   **Implementation:**
    1.  Audio frames are read directly into pre-allocated circular ring buffers in volatile system RAM (`/dev/shm` or pinned pinned CUDA memory).
    2.  Extracted mathematical tensors (spectrograms, phase vectors, LPC coefficients) are transformed into non-invertible embeddings.
    3.  Raw PCM audio packets are overwritten immediately as the pointer advances.
    4.  No audio byte is ever written to disk, cold storage, or external logs.

---

## 4. Multi-Stage Detection Pipeline

### Stage 1: Fast Steganographic Watermark Scanning
*   Scans frequency sub-bands for spread-spectrum watermarks embedded by commercial AI voice providers:
    *   **Google SynthID:** Psychoacoustic spread-spectrum phase/frequency markers.
    *   **Meta AudioSeal:** Sample-level localization watermark detector.
    *   **ElevenLabs Provenance Token:** High-frequency steganographic marker.
*   **Speed:** <10ms computation.
*   **Accuracy:** If a known watermark matches, confidence is **99.9% synthetic**, triggering an immediate alert.

### Stage 2: Generative Reconstructibility & Machine Regularities (Dissertation Method)
When scammers use open-source weights (XTTS, StyleTTS2, VITS, Bark, F5-TTS) with no watermarks, our physical acoustic inspection activates:

1.  **Neural Vocoder Phase Inconsistency:**
    *   Natural human speech production relies on continuous vocal tract physical geometry, producing smooth phase transitions.
    *   Neural vocoders (HiFi-GAN, BigVGAN, MelGAN) construct audio from 2D mel-spectrograms using upsampling transposed convolutions. This leaves subtle periodic phase discontinuities and harmonic phase mismatches across frequency bins, even after telephony compression (AMR-WB / G.711).
2.  **Higher-Order Spectral Statistics (Bispectrum & Bicoherence):**
    *   Human vocal cord vibration creates non-linear quadratic phase coupling (QPC) between fundamental pitch $F_0$ and higher formants.
    *   Synthetic voice algorithms over-regularize spectral envelopes, resulting in an unnaturally flat bicoherence surface.
3.  **Linear Predictive Coding (LPC) Glottal Residual Analysis:**
    *   Inverse filtering human speech using LPC removes vocal tract formant resonances, leaving the raw glottal excitation pulse.
    *   In authentic human speech, glottal pulses exhibit natural biological perturbations (micro-jitter and shimmer).
    *   In cloned AI speech, the residual exhibits characteristic neural vocoder artifacts: periodic high-frequency energy collapse and unnatural pulse symmetry.
4.  **Generative Inversion & Latent Reconstruction Error:**
    *   We test whether the observed acoustic features can be projected onto a low-dimensional manifold of known neural generative priors. If the reconstruction loss is exceptionally low under an inverse generator, the voice is deemed machine-generated.

### Stage 3: Noise-Based Acoustic Certificate & Dynamic Challenge-Response
*   **The Problem:** Future AI models might attempt to smooth out vocoder artifacts. How do we build an unbreakable cryptographic barrier?
*   **The Innovation:**
    *   Each subscriber is assigned a unique, unpredictable, noise-based acoustic certificate $\mathcal{C}_{sub}$ generated from a secure cryptographic seed.
    *   When an in-call situation is flagged as high-risk (e.g., call from an unknown number impersonating a relative, claiming an emergency, or requesting bank credentials), the network challenges the caller.
    *   The challenge introduces an acoustic micro-prompt or requires an instantaneous conversational exchange.
    *   **Why Generative Cloners Fail:**
        1.  *Latency Penalty:* Generative voice-cloning pipelines require at least 800–2000ms to transcribe inbound audio (ASR), generate an LLM response or operator script, and synthesize speech (TTS). They cannot maintain instantaneous conversational turn-taking.
        2.  *Acoustic Watermark Obliteration:* Even if an attacker attempts to replay or simulate the subscriber's voice, their generative neural network acts as a lossy non-linear filter that completely destroys the micro-acoustic certificate noise structure. The network easily detects the absence of the genuine physical acoustic token.

---

## 5. Zero-App Warning Mechanism

How does an elderly grandmother on a 15-year-old Nokia phone receive the warning?

1.  **In-Call Audio Whisper Injection (Primary):**
    *   The SBC media mixer momentarily attenuates the caller's audio by 12dB and injects a distinct, friendly carrier audio chime followed by a calm voice whisper into the subscriber's downlink channel:
        *   *"Notice: This voice sounds artificially generated. Do not send money or share bank passwords."*
    *   The scammer cannot hear this whisper (injected strictly into the callee's downlink stream).
2.  **Telco Class-0 Flash SMS / USSD Pop-up (Secondary):**
    *   Simultaneously, the telecom switch sends a high-priority Class-0 Flash SMS or USSD message that immediately overrides the phone screen, displaying:
        *   `⚠️ WARNING: AI Synthetic Voice Detected on Current Call. Hang up and call your contact directly.`
    *   Class-0 SMS appears immediately on all GSM phones without needing an app, requiring user acknowledgment to dismiss.
