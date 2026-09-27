import urllib.request
import json
import os

base_url = "http://127.0.0.1:8000"
boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW"

files = [
    ("User Real Voice", r"C:\Users\Grigor\Desktop\downloadq.wav"),
    ("Gemini AI Voice", r"C:\Users\Grigor\Desktop\download.wav")
]

print("================================================================================")
print("     TELEGUARD AI: END-TO-END HTTP FORENSIC VERIFICATION ON DESKTOP FILES       ")
print("================================================================================\n")

for label, file_path in files:
    fname = os.path.basename(file_path)
    with open(file_path, "rb") as f:
        wav_bytes = f.read()

    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{fname}"\r\n'
        f"Content-Type: audio/wav\r\n\r\n"
    ).encode("utf-8") + wav_bytes + f"\r\n--{boundary}--\r\n".encode("utf-8")

    req = urllib.request.Request(
        f"{base_url}/api/detect-audio-file",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"}
    )

    res = urllib.request.urlopen(req)
    d = json.loads(res.read())

    print(f"--- [{label.upper()}] ({fname}) ---")
    print(f"Verdict:            {d.get('verdict')}")
    print(f"Verdict Title:      {d.get('verdict_title')}")
    print(f"Confidence:         {d.get('confidence_pct')}%")
    print(f"Synthetic Prob:     {d.get('synthetic_probability')}")
    print(f"Is AI Generated:    {d.get('is_ai_generated')}")
    bio = d.get('biometrics', {})
    print(f"Comb Periodicity:   {bio.get('high_freq_comb_periodicity')}")
    print(f"Laryngeal Tremor:   {bio.get('laryngeal_tremor_pct')}%")
    print(f"RAP Jitter:         {bio.get('pitch_jitter_rap_pct')}%")
    print(f"Telecom Action:     {d.get('telecom_action')}")
    print(f"Summary:            {d.get('summary')}")
    print()
