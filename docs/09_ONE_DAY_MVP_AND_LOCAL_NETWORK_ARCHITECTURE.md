# 1-Day MVP Blueprint: Local Network Deployment & Rolling Acoustic Certificates

---

## 1. How to Implement the MVP in One Day (Without a Telecom Operator)

At an ideathon, you cannot obtain carrier core network access (IMS/SBC) in 24 hours. However, you can prove the **exact same in-network side-channel architecture** using a **Local Network / Edge Gateway**.

Here are three practical 1-day deployment models:

```
[MODEL A: The Hackathon Pitch Winner (Zero Hardware)]
=============================================================================
Caller Device (Phone / Laptop)               Victim Handset (Phone / Laptop)
       |                                                    ^
       | [Live WebRTC / WebSocket Audio Stream]             | [Audio + In-Call Whisper]
       v                                                    |
+-----------------------------------------------------------+----------------+
|  LOCAL EDGE GATEWAY (Running on Laptop or Raspberry Pi on Local Wi-Fi)     |
|  - Acts as local PBX / Media Relay (FreeSWITCH / WebRTC Stream Proxy)      |
|  - Bifurcates RTP stream into Ephemeral RAM Ring Buffer                    |
|  - Runs Real-Time Detector (<25ms)                                         |
|  - Mixes In-Call Whisper Tone into Victim's Downlink on AI Voice Detection |
+----------------------------------------------------------------------------+

[MODEL B: Home Landline / VoIP ATA Box (Hardware Demo)]
=============================================================================
Traditional Analog Landline Phone ──> Local ATA VoIP Adapter (Grandstream HT801)
                                              │
                                              ▼ (SIP / RTP over Local LAN)
                                      Local Python SBC Gateway
                                      (Pipes audio, runs detection, injects whisper)

[MODEL C: Wi-Fi Router / Hotspot Sidecar]
=============================================================================
Smartphone on Home Wi-Fi (VoWiFi Calling) ──> Local Router Packet Filter (OpenWrt / eBPF)
```

### Recommended Hackathon 1-Day MVP: Model A (Local WebRTC/WebSocket Media Gateway)
*   **Why it wins:** You can run it on your laptop in 10 minutes, open two browser windows or connect your phone via local Wi-Fi IP (`http://192.168.x.x:8000`), speak into the microphone live, and demonstrate instant synthetic voice detection and whisper tone injection.
*   **Zero App on the Victim Device:** The victim simply connects via any WebRTC phone or standard browser link without downloading an app.

---

## 2. Client-Specific Rolling Noise Certificates (Time-Varying Acoustic Watermarks)

### The Core Problem:
If an acoustic certificate is static, an attacker could record a phone call once, extract the noise pattern, and replay it.

### The Solution: Ephemeral Time-Rolling Acoustic Certificates ($\mathcal{C}_{client}(t)$)
Similar to **Time-Based One-Time Passwords (TOTP)** used in multi-factor authentication (Google Authenticator), the acoustic certificate **rotates automatically over time** based on a shared cryptographic seed and synchronized time epochs (e.g., rotating every 30 or 60 seconds).

```
                      CRYPTOGRAPHIC ROLLING CERTIFICATE PIPELINE
                      ==========================================

   [Client Identity]        [Master Secret / Private Seed]       [Current UTC Time Epoch]
     (e.g., Grandson)            (Derived at enrollment)           (Floor(t / 30 seconds))
            │                              │                                  │
            └──────────────────────────────┼──────────────────────────────────┘
                                           │
                                           ▼
                    HMAC-SHA256( Seed_Client , Epoch_Window_N )
                                           │
                                           ▼ [256-bit Cryptographic Hash]
                      Deterministic Pseudo-Random Generator (PRNG)
                                           │
                                           ▼
       128-point Psychoacoustic Noise Certificate Vector: C_client(t)
       - Frequency band: 2,500 Hz – 4,800 Hz (Sub-audible masked spectral shape)
       - Energy: -35dB relative to speech (imperceptible to human ear)
```

---

## 3. Why Generative AI Voice Cloners Mathematically Cannot Replicate This

| Property | Legitimate Human + Dynamic Certificate | Generative AI Voice Cloner (Attacker) |
| :--- | :--- | :--- |
| **High-Entropy Noise Reproduction** | Physical microphone / acoustic token directly emits $\mathcal{C}_{client}(t)$. | **Destruction by Vocoders:** Generative models (mel-spectrograms + HiFi-GAN) denoise and smooth out high-frequency micro-perturbations. |
| **Knowledge of the Secret Seed** | Stored securely on client hardware / enclave. | **Attacker Blindness:** Attacker only has a 3-second audio sample scraped from Instagram; they do not have the private seed. |
| **Time-Window Expiration** | Rotates every 30 seconds. Epoch $T$ expires at $T+1$. | **Replay Failure:** Even if an attacker records a call from yesterday, that certificate has expired and will be rejected instantly. |
| **Latency Budget** | Zero delay (instant acoustic response). | **Pipeline Lag:** Cloner incurs 800–2,000ms delay (ASR $\rightarrow$ LLM $\rightarrow$ TTS), failing interactive challenge timing. |

---

## 4. Verification Flow: How the Network Validates the Rolling Certificate

```
Active Call Ongoing
       │
[Suspicion Triggered by Tier 2 (Vocoder Artifacts > 0.65)]
       │
       ▼
System queries: "Who does the caller claim to be?"
       │ Claimed Identity = "Grandson (ID: #48102)"
       ▼
System fetches Grandson's registered Public Master Key
Computes Expected Rolling Certificate:
       C_expected = DeriveCertificate(Seed_Grandson, Current_Epoch_Time)
       │
Extracts residual high-frequency spectrum from incoming caller audio:
       S_caller = Bandpass_FFT(Audio_Chunk, 2500Hz - 4800Hz)
       │
Computes Normalized Cross-Correlation:
       Score = DotProduct( Normalized(S_caller) , Normalized(C_expected) )
       │
       ├── If Score >= Threshold (0.45) ──> VALID AUTHENTICATED CALLER
       │
       └── If Score < Threshold (0.45)  ──> CRYPTOGRAPHIC CERTIFICATE MISMATCH
                                             --> AI Voice Cloner Detected!
                                             --> Inject In-Call Whisper Alert!
```
