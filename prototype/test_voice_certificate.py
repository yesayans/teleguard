"""
Test Suite: Voice Certificate Module
Verifies:
1. Genuine Certified Caller (Correct secret key, active 30s epoch, matching call_id) -> PASS
2. AI Voice Clone (Copies voice timbre, lacks secret key) -> BLOCKED
3. Replay Attack (Scammer replays genuine audio from expired epoch) -> BLOCKED (REPLAY DETECTED)
4. Random Noise Injection (Attacker attempts to fake certificate with white noise) -> BLOCKED
"""

import os
import sys
import time
import numpy as np

# Ensure root workspace is on python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from prototype.core.voice_certificate import VoiceCertificateEmbedder, VoiceCertificateVerifier
from prototype.run_demo import generate_human_voice_sample, generate_synthetic_voice_sample

def run_tests():
    print("=" * 78)
    print(" VOICE CERTIFICATE MODULE: AUTOMATED VERIFICATION TEST SUITE")
    print(" Testing Cryptographic Dynamic Acoustic Watermarking & Replay Protection")
    print("=" * 78)

    SECRET_KEY = "GRANDSON_PRIVATE_KEY_9841"
    CALL_ID = "CALL_2026_09_27_ARM_001"
    
    embedder = VoiceCertificateEmbedder(secret_key=SECRET_KEY, epoch_seconds=30)
    verifier = VoiceCertificateVerifier(secret_key=SECRET_KEY, epoch_seconds=30)
    
    base_voice = generate_human_voice_sample(duration_sec=0.4)
    current_time = time.time()

    # --- TEST 1: GENUINE CERTIFIED CALLER ---
    print("\n[TEST 1] Genuine Certified Caller (Sending Side Embeds Dynamic Token)...")
    certified_audio = embedder.embed_certificate(base_voice, call_id=CALL_ID, timestamp=current_time)
    res1 = verifier.verify_frame(certified_audio, call_id=CALL_ID, timestamp=current_time)
    print(f" -> Result: {res1['status']}")
    print(f" -> Correlation Score: {res1['correlation_score']:.4f} (Threshold: 0.40)")
    print(f" -> Reason: {res1['reason']}")
    assert res1["verified"] is True, "Test 1 Failed: Legitimate certificate should verify!"
    print(" [OK] TEST 1 PASSED: Authentic caller verified successfully.")

    # --- TEST 2: AI VOICE CLONE (XTTS / ElevenLabs) ---
    print("\n[TEST 2] AI Voice Clone (Attacker clones voice but lacks private secret key)...")
    ai_cloned_voice = generate_synthetic_voice_sample(duration_sec=0.4)
    # The cloner synthesizes the voice, but neural vocoder has no certificate
    res2 = verifier.verify_frame(ai_cloned_voice, call_id=CALL_ID, timestamp=current_time)
    print(f" -> Result: {res2['status']}")
    print(f" -> Correlation Score: {res2['correlation_score']:.4f}")
    print(f" -> Reason: {res2['reason']}")
    assert res2["verified"] is False, "Test 2 Failed: AI Voice Clone must be rejected!"
    print(" [OK] TEST 2 PASSED: AI voice clone correctly blocked (Certificate Missing).")

    # --- TEST 3: REPLAY ATTACK (Scammer replays old recorded call) ---
    print("\n[TEST 3] Replay Attack (Scammer replays genuine call recorded from an old epoch)...")
    old_timestamp = current_time - 150 # 5 epochs ago (2.5 minutes ago)
    replayed_audio = embedder.embed_certificate(base_voice, call_id=CALL_ID, timestamp=old_timestamp)
    # Target receives replayed audio during current time epoch
    res3 = verifier.verify_frame(replayed_audio, call_id=CALL_ID, timestamp=current_time)
    print(f" -> Result: {res3['status']}")
    print(f" -> Correlation Score (Active Epoch): {res3['correlation_score']:.4f}")
    print(f" -> Replay Attack Flagged: {res3['replay_attack']}")
    print(f" -> Reason: {res3['reason']}")
    assert res3["verified"] is False, "Test 3 Failed: Replayed call must be rejected!"
    print(" [OK] TEST 3 PASSED: Replay attack detected and blocked.")

    # --- TEST 4: RANDOM NOISE SPOOFING (Attacker injects arbitrary noise) ---
    print("\n[TEST 4] Random Noise Spoofing (Attacker adds loud white noise hoping to fool detector)...")
    noise_spoofed_audio = base_voice + np.random.normal(0.0, 0.05, len(base_voice))
    res4 = verifier.verify_frame(noise_spoofed_audio, call_id=CALL_ID, timestamp=current_time)
    print(f" -> Result: {res4['status']}")
    print(f" -> Correlation Score: {res4['correlation_score']:.4f}")
    assert res4["verified"] is False, "Test 4 Failed: Random noise should have near-zero correlation!"
    print(" [OK] TEST 4 PASSED: Random noise injection blocked.")

    print("\n" + "=" * 78)
    print(" ALL VOICE CERTIFICATE TESTS PASSED (4/4)")
    print(" Cryptographic Guarantee: AI Cloners cannot replicate unshared rotating secrets.")
    print("=" * 78)

if __name__ == "__main__":
    run_tests()
