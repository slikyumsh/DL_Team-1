import os

AUDIO_DIR = 'artifacts/audio'
OUTPUT_HTML = 'artifacts/plots/mos_rating.html'

os.makedirs(os.path.dirname(OUTPUT_HTML), exist_ok=True)
audio_files = sorted([f for f in os.listdir(AUDIO_DIR) if f.endswith('.wav')])

html = "<html><head><title>MOS Rating</title></head><body>"
html += "<h1>Mean Opinion Score (MOS) — Synthetic Sounds</h1>"
html += "<form>"

for i, filename in enumerate(audio_files):
    html += f"<h3>Sound {i+1}</h3>"
    html += f"<audio controls src='../audio/{filename}'></audio><br>"
    html += f"<label>Rate quality (1 = Bad, 5 = Excellent):</label><br>"
    for score in range(1, 6):
        html += f"<input type='radio' name='sound{i}' value='{score}'> {score} "
    html += "<br><hr>"

html += "</form></body></html>"

with open(OUTPUT_HTML, 'w') as f:
    f.write(html)

print(f"✅ HTML для MOS сохранён: {OUTPUT_HTML}")
