# Milestone 1: Problem and Solution Definition

## 1. Locked Problem Statement (Ideathon Criteria: Real, Specific, Plausible)

### The One-Sentence Definition:
> **"Criminals weaponize 3-second generative voice cloning to defraud vulnerable telecom subscribers—especially the elderly—causing over $3.5B in annual imposter losses, while existing defenses fail because 94% of elderly victims cannot or will not install security applications."**

---

## 2. Who Has This Problem?

| Stakeholder | Pain Point | Impact / Severity |
| :--- | :--- | :--- |
| **Vulnerable Subscribers (Elderly & Families)** | Cannot distinguish cloned voices of grandchildren, children, or bank officials; targeted by emotional extortion ("grandparent scam", hospital emergencies, fake arrests). | Life savings wiped out ($3.5B imposter losses in 2025; avg. loss >$2,000 per senior victim; 51% loss rate on AI-assisted attacks). |
| **Adult Children / Caregivers** | Extreme anxiety regarding parents' vulnerability; unable to supervise unrecorded live phone calls on landlines or basic feature phones. | Willing to pay for automated network protection to safeguard parents. |
| **Telecom Operators (Telcos / MNOs)** | Severe erosion of caller trust (subscribers stop answering unknown calls; voice traffic degradation); commoditized voice revenues with zero differentiation. | Regulatory pressure (FCC, EU telecom mandates) to combat spoofing/fraud; missed opportunity for high-margin cybersecurity VAS (Value Added Services). |
| **Banks & FinTechs** | Authorized Push Payment (APP) fraud; victims are coerced into initiating transfers themselves, making standard transaction-monitoring rules ineffective. | Reimbursement liability under emerging regulations (e.g., UK PSR mandatory 50/50 refund split; upcoming EU Payment Services Regulations). |

---

## 3. Why Now? (Technological & Market Inflection Point)

1. **Near-Zero Barrier to Voice Cloning (2024–2026):**
   - Modern open-source models (XTTS-v2, F5-TTS, StyleTTS2, VITS, Kokoro) and commercial APIs (ElevenLabs, Cartesia) require as little as **3 seconds of audio** scraped from social media (TikTok, Instagram, LinkedIn, voicemail).
   - Cost to attackers: **<$0.005 per cloned call**, executable from consumer laptops or cheap cloud VMs.
2. **Human Perceptual Blindness:**
   - Independent studies (McAfee, University College London) reveal that **70% to 77% of adults cannot reliably distinguish AI-generated synthetic voices from authentic human speech**. Under emotional duress (e.g., a "crying grandchild"), discrimination drops to near zero.
3. **The App-Store Blindspot:**
   - Traditional endpoint security (mobile apps, browser extensions, antivirus) completely misses the most vulnerable demographic:
     - 68% of people aged 70+ do not download new security apps.
     - Millions still use basic feature phones, 2G/3G legacy handsets, or standard fixed landlines.
     - iOS and Android sandboxing prevents third-party apps from inspecting live cellular voice RTP audio streams due to platform security restrictions.
4. **Regulatory Impetus:**
   - FTC Voice Cloning Challenge, FCC mandates expanding beyond STIR/SHAKEN caller ID spoofing into audio content authentication, and EU AI Act deepfake disclosure requirements.

---

## 4. The Solution: In-Network Synthetic Voice Detection & Acoustic Attestation

### Core Value Proposition:
**"Zero-App, Zero-Device-Constraint, In-Network Deepfake Voice Defense"**
A telecom-grade infrastructure solution running natively within the mobile network operator's (MNO) core infrastructure (IMS / Session Border Controller / Kamailio / FreeSWITCH). It inspects inbound voice audio streams in volatile RAM ring buffers with zero audio recording, providing sub-second warnings directly into the ongoing call.

```
                    TELECOM OPERATOR CORE NETWORK (IMS / SBC)
                    ==========================================
Caller (Attacker)                                                  Subscriber (Grandmother)
    |                                                                         ^
    | [RTP Voice Stream]                                                      |
    v                                                                         |
+-----------------------------------------------------------------------------+
| Session Border Controller (SBC) / B2BUA                                     |
|   |--> RTP Stream Bifurcated (Zero-Delay Side-Channel Tap)                  |
|   |                                                                         |
|   v                                                                         |
| [VOLATILE MEMORY RING BUFFER] (100ms Chunks - Stored in RAM, Never Disk)    |
|   |                                                                         |
|   +--> Stage 1: Fast Audio Watermark Scanner (SynthID, AudioSeal, ElevenLabs)
|   |            | Found: 99.9% Confidence Synthetic -> Alert                 |
|   |            | Not Found: Pass to Stage 2                                 |
|   |                                                                         |
|   +--> Stage 2: Generative Reconstructibility & Machine Regularity Analyzer|
|   |            | - Phase Spectrum Discontinuity (Vocoder Artifacts)         |
|   |            | - Bispectral Bicoherence & Glottal Pulse LPC Residual      |
|   |            | - Machine Latent Regularity Reconstruction Error           |
|   |                                                                         |
|   +--> Stage 3: Noise-Based Acoustic Certificate & Dynamic Challenge        |
|                | Suspicious score > threshold?                              |
|                | In-call acoustic challenge with subscriber's unique token  |
+-----------------------------------------------------------------------------+
    |
    v
[REAL-TIME IN-CALL INTERVENTION] (Zero App Required)
    --> In-band gentle audio whisper tone: "Security Alert: Synthetic Voice Detected"
    --> Immediate Telco Flash-SMS / USSD screen takeover: "⚠️ Caution: High probability of AI Voice Cloning"
```

---

## 5. Technical Novelty & Breakthroughs

1. **In-Network Passive Inspection (No App, No Hardware):**
   - Works on every single phone: $15 Nokia 105 feature phone, PSTN landline, iPhone 16 Pro, or Samsung Galaxy.
   - Subscriber needs to do **nothing**. Protection is enabled automatically at the SIM/carrier level.
2. **Triple-Layered Detection Engine:**
   - **Layer 1 (Known Providers):** Rapid parsing for steganographic spread-spectrum watermarks (Google SynthID, Meta AudioSeal, ElevenLabs provenance).
   - **Layer 2 (Open-Source / Dissertation Method):** Analysis of generative reconstructibility. Models like VITS, StyleTTS2, and Bark exhibit spectral phase phase-wrapping anomalies, artificial glottal closure moments in LPC residuals, and higher-order spectral statistics (bispectrum) that human vocal cords cannot physically produce across telephony codecs (AMR-WB / AMR-NB / G.711).
   - **Layer 3 (Noise-Based Acoustic Certificate & Challenge):** Each subscriber is provisioned with a cryptographic, noise-based acoustic certificate. If a high-stakes call (e.g. wire transfer, bail money) occurs, an unpredictable acoustic challenge-response verified through the network exposes generative voice clones that cannot simulate instantaneous physical acoustic perturbations.
3. **Absolute Privacy Compliance (Zero-Recording Guarantee):**
   - Audio is buffered strictly in ephemeral RAM ring buffers for sliding 200–500ms DSP analysis windows.
   - Vectors/embeddings are classified; raw audio is immediately overwritten. Zero disk I/O, zero audio wiretapping, 100% compliant with GDPR Art. 5/6, ePrivacy Directive, and US Title III wiretap statutes.
4. **Low-Resource Language Resilience:**
   - Focuses on **fundamental physical acoustics and vocoder phase artifacts**, rather than semantic/lexical meaning. Works with equal accuracy in English, Armenian, Spanish, German, Hindi, or Arabic.
