import urllib.request
import json
import numpy as np
import scipy.signal
import io
import soundfile as sf

base_url = 'http://127.0.0.1:8000'

def test_preset(preset_name):
    req = urllib.request.Request(f'{base_url}/api/analyze-preset',
                                 data=json.dumps({'preset': preset_name}).encode('utf-8'),
                                 headers={'Content-Type': 'application/json'})
    res = urllib.request.urlopen(req)
    d = json.loads(res.read())
    verdict = d.get('verdict', 'UNKNOWN')
    prob = d.get('synthetic_probability', 0.0)
    b = d.get('biometrics', {})
    tremor = b.get('laryngeal_tremor_pct', 0.0)
    comb = b.get('high_freq_comb_periodicity', 0.0)
    print(f"Preset [{preset_name:12s}] -> Verdict: {verdict:22s} | Prob: {prob:6.4f} | Tremor: {tremor:5.3f}% | Comb: {comb:5.3f}")

print('--- PRESET BENCHMARKS ---')
test_preset('human')
test_preset('human_noisy')
test_preset('ai_clone')
test_preset('ai_noisy')

# Now test Laptop Sound ElevenLabs AI audio via /api/analyze-recording
fs = 16000
duration = 2.5
t = np.linspace(0, duration, int(fs * duration), endpoint=False)
f0_ai = 135.0 + 20.0 * np.sin(2 * np.pi * 0.7 * t) + 10.0 * np.cos(2 * np.pi * 1.5 * t)
phase_ai = 2 * np.pi * np.cumsum(f0_ai) / fs
sig_ai = np.zeros_like(t)
for h in range(1, 24):
    sig_ai += (1.0 / (h ** 0.85)) * np.sin(h * phase_ai)
env_words = np.clip(0.6 + 0.4 * np.sin(2 * np.pi * 2.0 * t), 0.05, 1.0)
sig_ai *= env_words
sig_ai /= np.max(np.abs(sig_ai))

# Laptop sound simulation
b_hp, a_hp = scipy.signal.butter(2, 175 / (fs/2), btype='high')
laptop_ai = scipy.signal.lfilter(b_hp, a_hp, sig_ai)
d1, d2 = int(fs * 0.0015), int(fs * 0.005)
laptop_ai[d1:] += 0.30 * laptop_ai[:-d1]
laptop_ai[d2:] += 0.15 * laptop_ai[:-d2]
laptop_ai += np.random.normal(0, 0.025, len(laptop_ai))
laptop_ai /= np.max(np.abs(laptop_ai))

req = urllib.request.Request(f'{base_url}/api/analyze-recording',
                             data=json.dumps({'samples': laptop_ai.tolist(), 'client_sample_rate': 16000}).encode('utf-8'),
                             headers={'Content-Type': 'application/json'})
res = urllib.request.urlopen(req)
d_laptop = json.loads(res.read())
b_lap = d_laptop.get('biometrics', {})
print('\n--- LAPTOP SOUND AI VOICE TEST ---')
print(f"Laptop Sound AI Voice -> Verdict: {d_laptop.get('verdict', ''):22s} | Prob: {d_laptop.get('synthetic_probability', 0):6.4f} | Tremor: {b_lap.get('laryngeal_tremor_pct', 0):5.3f}% | Comb: {b_lap.get('high_freq_comb_periodicity', 0):5.3f}")

# Test MP3 upload via /api/detect-audio-file
buf = io.BytesIO()
sf.write(buf, laptop_ai, fs, format='MP3')
mp3_bytes = buf.getvalue()

boundary = '----WebKitFormBoundary7MA4YWxkTrZu0gW'
body = (
    f'--{boundary}\r\n'
    f'Content-Disposition: form-data; name="file"; filename="test_laptop_ai.mp3"\r\n'
    f'Content-Type: audio/mpeg\r\n\r\n'
).encode('utf-8') + mp3_bytes + f'\r\n--{boundary}--\r\n'.encode('utf-8')

req_mp3 = urllib.request.Request(f'{base_url}/api/detect-audio-file',
                                 data=body,
                                 headers={'Content-Type': f'multipart/form-data; boundary={boundary}'})
res_mp3 = urllib.request.urlopen(req_mp3)
d_mp3 = json.loads(res_mp3.read())
b_mp3 = d_mp3.get('biometrics', {})
print('\n--- MP3 FILE UPLOAD TEST ---')
print(f"Uploaded MP3 File     -> Verdict: {d_mp3.get('verdict', ''):22s} | Prob: {d_mp3.get('synthetic_probability', 0):6.4f} | Tremor: {b_mp3.get('laryngeal_tremor_pct', 0):5.3f}% | Comb: {b_mp3.get('high_freq_comb_periodicity', 0):5.3f}")
