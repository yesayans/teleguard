import urllib.request
import json

base_url = "http://127.0.0.1:8000"
boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW"

with open(r"C:\Users\Grigor\Desktop\download.wav", "rb") as f:
    wav_bytes = f.read()

body = (
    f"--{boundary}\r\n"
    f'Content-Disposition: form-data; name="file"; filename="download.wav"\r\n'
    f"Content-Type: audio/wav\r\n\r\n"
).encode("utf-8") + wav_bytes + f"\r\n--{boundary}--\r\n".encode("utf-8")

req = urllib.request.Request(
    f"{base_url}/api/detect-audio-file",
    data=body,
    headers={"Content-Type": f"multipart/form-data; boundary={boundary}"}
)

res = urllib.request.urlopen(req)
d = json.loads(res.read())

print("=== VERIFICATION ON DESKTOP FILE (download.wav) ===")
print("Verdict:            ", d.get("verdict"))
print("Verdict Title:      ", d.get("verdict_title"))
print("Confidence:         ", d.get("confidence_pct"), "%")
print("Synthetic Prob:     ", d.get("synthetic_probability"))
print("Is AI Generated:    ", d.get("is_ai_generated"))
print("Comb Periodicity:   ", d.get("biometrics", {}).get("high_freq_comb_periodicity"))
print("Laryngeal Tremor:   ", d.get("biometrics", {}).get("laryngeal_tremor_pct"), "%")
print("RAP Jitter:         ", d.get("biometrics", {}).get("pitch_jitter_rap_pct"), "%")
print("Telecom Action:     ", d.get("telecom_action"))
print("Summary:            ", d.get("summary"))
