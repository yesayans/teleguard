# Milestone 1: Market Sizing, Economics & Pricing Strategy

## 1. Market Opportunity: Why Voice Security is the Next Multi-Billion Frontier

Telecom operators are experiencing historic declines in voice service value: subscribers distrust calls from unknown numbers, call completion rates are down 42%, and imposter scams have exploded. Simultaneously, telcos are desperate for **high-margin Value-Added Services (VAS)** to differentiate beyond commoditized 5G data pipes.

---

## 2. Market Sizing (TAM, SAM, SOM)

```
+--------------------------------------------------------------------------+
|  TAM: $28.5 Billion                                                      |
|  Global Telecom Voice Security & Consumer Identity Protection VAS        |
|  (5.6B mobile subscribers worldwide; $0.50-$2.00/mo VAS potential)        |
+--------------------------------------------------------------------------+
       |
       v
+-------------------------------------------------------------------+
|  SAM: $2.1 Billion                                                |
|  Targetable MNO Subscribers in Europe, North America & High-Risk   |
|  Regions with Modern IMS/VoLTE (Senior & Family Plan Segments)    |
|  (~500M addressable lines @ $0.35/mo wholesale)                   |
+-------------------------------------------------------------------+
       |
       v
+------------------------------------------------------------+
|  SOM: $19.5 Million ARR (Years 1–3)                        |
|  3–5 Regional MNO Pilots + Tier-2 Challengers in Europe/   |
|  Caucasus (Target: 6.5M protected subscribers by Month 36) |
+------------------------------------------------------------+
```

### Detailed Breakdown:

### A. Total Addressable Market (TAM): **$28.5 Billion**
*   **Methodology:** 
    *   According to GSMA Intelligence, there are **5.6 billion unique mobile subscribers** globally.
    *   The telecom cybersecurity & fraud prevention market is projected to reach **$18.2B by 2028** (CAGR 16.4%).
    *   Global consumer fraud losses surpassed **$16B in 2025** (FTC: $3.5B imposter scams in US alone; UK PSR: £460M in APP fraud; EU and APAC proportional).
    *   If 25% of global subscribers adopt or are covered by automated carrier-level fraud filtering at an average retail price of $1.50/month: $1.50 \times 12 \times 1.4\text{B subscribers} \approx \$25.2\text{B}$ consumer market + $\$3.3\text{B}$ carrier infrastructure software.

### B. Serviceable Addressable Market (SAM): **$2.1 Billion (Wholesale)**
*   **Target Scope:** Telecom operators across the EU, UK, North America, and Eastern Europe/Caucasus that have migrated to IMS/VoLTE and face strict regulatory pressure (EU AI Act, FCC Robocall/Anti-Spoofing regulations, UK PSR fraud reimbursement mandates).
*   **Target Population:** 500 million active subscribers across these regulated markets, focusing on senior households, family plans, and high-net-worth mobile subscribers.
*   **Wholesale Value:** $500\text{M lines} \times \$0.35\text{/month wholesale} \times 12 = \$2.1\text{B ARR}$.

### C. Serviceable Obtainable Market (SOM): **$19.5 Million ARR (Year 3)**
*   **Beachhead Strategy:** 
    *   **Phase 1 (Months 1–12):** Launch pilot in Armenia / South Caucasus (Viva, Telecom Armenia, Ucom – ~3.8M subscriber base) to validate zero-latency deployment on a live MNO network.
    *   **Phase 2 (Months 12–24):** Expand to 2–3 challenger telcos in Eastern/Southern Europe (e.g., Iliad, Digi, Three). Reach 1.5M protected lines ($4.5M ARR).
    *   **Phase 3 (Months 24–36):** Scale to Tier-1 European carrier group (e.g., Orange, Vodafone, or Deutsche Telekom ecosystem). Reach 6.5M protected lines ($19.5M ARR).

---

## 3. Revenue Models & Pricing Strategy

We employ a **hybrid B2B / B2B2C licensing model** engineered specifically for telecom procurement cycles:

### Tier 1: Per-Subscriber-Per-Month (PSPM) Wholesale Licensing (Core Model)
*   **Wholesale Price to Telco:** **$0.20 – $0.40 / subscriber / month** (tiered based on volume).
*   **Telco Go-To-Market Options:**
    *   *Option A (Premium Add-On):* Telco sells to subscribers as **"SafeVoice AI Shield"** for **$1.99 – $2.99 / month**. Telco keeps an **80–85% gross margin**.
    *   *Option B (Family / Senior Plan Bundle):* Telco bundles it into family plans or senior packages ($15–$30/mo) at zero added friction to reduce churn by an estimated 15–22%.
    *   *Option C (Universal Default Tier):* Basic watermark check provided free to all users; deep spectral analysis + noise-certificate challenge reserved for premium tier.

### Tier 2: Per-Minute RTP Stream Inspection (Usage-Based)
*   **Pricing:** **$0.0025 – $0.0040 per minute** of inspected inbound audio.
*   **Target:** Wholesale carrier transit, international incoming gateways, and VoIP aggregators who want to scrub toxic traffic before handoff.

### Tier 3: FinTech & Banking API Co-Sponsorship (B2B Bounty / Risk API)
*   **Mechanism:** When a bank customer is on a call while attempting an instant transfer, the bank’s mobile app queries our Telco API via GSMA Open Gateway / CAMARA standards (`VerifyCallSafety` endpoint).
*   **Pricing:** **$0.08 – $0.15 per verification query**.
*   **ROI for Bank:** Preventing a single $5,000 imposter transfer pays for over 35,000 API checks.

---

## 4. Unit Economics & Gross Margins

```
Cost to Inspect 1 Minute of Telecom Audio:
------------------------------------------
Compute (Edge GPU / Optimized TensorRT/ONNX on CPU):      $0.00035
Telco SIP SBC Bifurcation Overhead:                      $0.00010
Infrastructure & Logging (Ephemeral Ring Buffer):         $0.00005
------------------------------------------------------------------
TOTAL COGS PER MINUTE:                                    $0.00050

Average User Profile:
- Inbound external calls inspected: 45 minutes / month
- Total Monthly COGS per Active Subscriber:              $0.0225

Wholesale Revenue per Subscriber:                        $0.3000
Gross Margin:                                            92.5%
```

With a COGS of ~$0.02 per user per month and wholesale revenue of $0.30, our software provides a **92.5% gross margin**, leaving immense room for telco integration support, R&D, and edge server scaling.
