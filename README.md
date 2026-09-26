# TeleGuard AI — In-Network Deepfake Voice Defense

> **Ideathon Submission & Research Package**  
> **Problem:** Criminals weaponize 3-second generative voice cloning to defraud vulnerable telecom subscribers (especially the elderly), causing $3.5B+ in annual imposter losses while app-based security solutions fail.  
> **Task:** Detect synthetic speech during a live phone call and warn the subscriber in real-time without requiring any app or user action.  
> **Submission Target:** `eutumocc@tumo.org` (EU TUMO Convergence Center)

---

## Complete Project Documentation & Research Assets

| Document | Description & Milestone Focus |
| :--- | :--- |
| **[01. Problem & Solution Definition](file:///c:/Users/Grigor/Documents/antigravity/friendly-kepler/docs/01_PROBLEM_AND_SOLUTION_DEFINITION.md)** | Locked one-sentence problem, target stakeholders, market inflection point, and zero-app solution architecture. |
| **[02. Competitor Research & Benchmarking](file:///c:/Users/Grigor/Documents/antigravity/friendly-kepler/docs/02_COMPETITOR_RESEARCH_AND_BENCHMARKING.md)** | Detailed matrix vs Pindrop, Reality Defender, Hiya, Truecaller, and ElevenLabs; analysis of the "app fallacy". |
| **[03. Market Sizing & Pricing Models](file:///c:/Users/Grigor/Documents/antigravity/friendly-kepler/docs/03_MARKET_SIZING_AND_PRICING_MODELS.md)** | Bottom-up TAM ($28.5B), SAM ($2.1B), SOM ($19.5M ARR); wholesale $0.30/sub/mo vs retail $1.99/mo; unit economics (92.5% gross margin). |
| **[04. Business Model Canvas](file:///c:/Users/Grigor/Documents/antigravity/friendly-kepler/docs/04_BUSINESS_MODEL_CANVAS.md)** | Complete 9-box Osterwalder Canvas tailored to telecom operators and banking API integrations. |
| **[05. Traction Evidence & Interview Scripts](file:///c:/Users/Grigor/Documents/antigravity/friendly-kepler/docs/05_TRACTION_RESEARCH_AND_INTERVIEW_SCRIPTS.md)** | Hard empirical sources (FTC, McAfee, UK PSR, GSMA); interview scripts and validation tally for telcos, banks, and senior caregivers. |
| **[06. Technical Architecture & Innovation](file:///c:/Users/Grigor/Documents/antigravity/friendly-kepler/docs/06_TECHNICAL_ARCHITECTURE_AND_INNOVATION.md)** | SIP/RTP SBC stream bifurcation, ephemeral volatile RAM ring buffers (GDPR compliant), dissertation vocoder reconstruction physics, and noise-based acoustic certificates. |
| **[07. Financials, Investment & Roadmap](file:///c:/Users/Grigor/Documents/antigravity/friendly-kepler/docs/07_FINANCIALS_INVESTMENT_AND_ROADMAP.md)** | $650,000 seed ask, 18-month use of funds, 3-year pro-forma P&L, team division, and operational milestones. |
| **[08. 5-Minute Pitch Deck & Script](file:///c:/Users/Grigor/Documents/antigravity/friendly-kepler/docs/08_PITCH_DECK_AND_SCRIPT_5MIN.md)** | Slide-by-slide 5-minute presentation script with stopwatch timings, judges' Q&A defense, and submission email draft. |

---

## Working Prototype & Live Demonstration

A complete, functioning simulation of the in-network telecom inspection pipeline is included in [`prototype/`](file:///c:/Users/Grigor/Documents/antigravity/friendly-kepler/prototype/):

### Key Components:
*   **[`core/watermark.py`](file:///c:/Users/Grigor/Documents/antigravity/friendly-kepler/prototype/core/watermark.py):** Tier 1 Steganographic spread-spectrum scanner (Google SynthID, Meta AudioSeal, ElevenLabs).
*   **[`core/detector.py`](file:///c:/Users/Grigor/Documents/antigravity/friendly-kepler/prototype/core/detector.py):** Tier 2 Dissertation Method analyzing vocoder phase discontinuities, LPC glottal excitation residuals, and bispectral bicoherence.
*   **[`core/certificate.py`](file:///c:/Users/Grigor/Documents/antigravity/friendly-kepler/prototype/core/certificate.py):** Tier 3 Subscriber-unique noise-based acoustic certificate & dynamic challenge-response engine.
*   **[`core/stream_simulator.py`](file:///c:/Users/Grigor/Documents/antigravity/friendly-kepler/prototype/core/stream_simulator.py):** In-network Session Border Controller (SBC) side-channel tap simulation with volatile RAM circular buffer (0 bytes written to disk).

### How to Run:

1. **Run the Automated Benchmark (CLI):**
   ```bash
   python prototype/run_demo.py
   ```
   *Demonstrates human voice pass-through vs. instantaneous detection and alert injection for open-source AI cloned voices (<7ms average frame latency).*

2. **Launch the Interactive Live Dashboard (Web UI):**
   ```bash
   python prototype/server.py
   ```
   *Open `http://127.0.0.1:8000` in any browser to interact with live waveform visualizations, latency metrics, threat gauges, and simulated handset in-call whisper alerts.*

3. **Present the Interactive Slide Deck:**
   *Open [`presentation/slides.html`](file:///c:/Users/Grigor/Documents/antigravity/friendly-kepler/presentation/slides.html) in your browser. Use Arrow Keys or Spacebar to navigate the 10 pitch slides during rehearsals.*
