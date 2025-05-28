import os
import random
from pathlib import Path
from dotenv import load_dotenv

"""
Генерирует HTML‑страницу для субъективной оценки (MOS) сгенерированных
звуков. Каждому плееру — радиогруппа 1…5. Итоговый файл можно открыть
локально в браузере или задеплоить на GitHub‑Pages.
"""

load_dotenv()

AUDIO_DIR = Path(os.getenv("GEN_AUDIO_DIR", "artifacts/audio"))
HTML_FILE = Path(os.getenv("MOS_HTML_PATH", "artifacts/plots/mos_rating.html"))

AUDIO_DIR.mkdir(parents=True, exist_ok=True)
HTML_FILE.parent.mkdir(parents=True, exist_ok=True)

audio_files = sorted([p for p in AUDIO_DIR.glob("*.wav")])
if not audio_files:
    raise SystemExit(f"Нет WAV‑файлов в {AUDIO_DIR}")

random.shuffle(audio_files)

rel_paths = [os.path.relpath(p, HTML_FILE.parent) for p in audio_files]

def html_radio(group: str) -> str:
    return " ".join(
        f"<label><input type='radio' name='{group}' value='{s}'> {s}</label>" for s in range(1, 6)
    )

html = [
    "<html><head><meta charset='utf-8'><title>MOS Rating</title></head><body>",
    "<h1 style='font-family:sans-serif'>Mean Opinion Score (MOS) ‒ Synthetic Sounds</h1>",
    "<p>Прослушайте каждый звук и отметьте субъективное качество (1 = плохо, 5 = отлично).</p>",
    "<form id='mosForm'>",
]

for idx, rel in enumerate(rel_paths, 1):
    html.append(f"<h3>Звук {idx}</h3>")
    html.append(f"<audio controls src='{rel}'></audio><br>")
    html.append(html_radio(f"sound{idx}"))
    html.append("<hr>")

html.extend([
    "<button type='submit'>Сохранить результаты</button>",
    "</form>",
    "</body></html>",
])

HTML_FILE.write_text("\n".join(html), encoding="utf-8")
print("HTML для MOS сохранён:", HTML_FILE)