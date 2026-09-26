"""ML 生成冒烟验证：从 val 集取一首真歌，检查长音/列分布/密度是否合理"""
import glob
import os
import sys
import tempfile
import zipfile

import numpy as np

from mustaff.analyzer import AudioAnalyzer
from training.generate import generate_notes_ml, models_available

STARS = float(sys.argv[1]) if len(sys.argv) > 1 else 6.0
HOLD_BIAS = float(sys.argv[2]) if len(sys.argv) > 2 else 2.0

print("models_available:", models_available())

osz = sorted(glob.glob("training/data/osz/*.osz"))[5]
with zipfile.ZipFile(osz) as zf:
    audio = [n for n in zf.namelist() if os.path.splitext(n)[1].lower() in (".mp3", ".ogg", ".wav")]
    biggest = max(audio, key=lambda n: zf.getinfo(n).file_size)
    data = zf.read(biggest)
print("测试歌曲:", os.path.basename(osz), "/", biggest)

with tempfile.NamedTemporaryFile(suffix=os.path.splitext(biggest)[1], delete=False) as t:
    t.write(data)
    tmp = t.name
try:
    y = AudioAnalyzer().load(tmp).y
finally:
    os.remove(tmp)

notes = generate_notes_ml(y, stars=STARS, head_thr=HOLD_BIAS)
holds = [n for n in notes if n["type"] == "hold"]
cols = [n["column"] for n in notes]
print(f"stars={STARS}  notes={len(notes)}  holds={len(holds)}  "
      f"hold_ratio={len(holds)/max(1,len(notes)):.3f}")
print("列分布:", np.bincount(cols, minlength=4).tolist())
if holds:
    durs = [n["end_time"] - n["time"] for n in holds]
    print(f"长音时长 ms: p10={np.percentile(durs,10):.0f} p50={np.percentile(durs,50):.0f} "
          f"p90={np.percentile(durs,90):.0f} max={max(durs)}")
jacks = sum(1 for a, b in zip(notes, notes[1:])
            if a["column"] == b["column"] and b["time"] - a["time"] < 100)
print("100ms 内同列 jack:", jacks)
from collections import Counter
chords = sum(1 for t, c in Counter(n["time"] for n in notes).items() if c >= 2)
print("和弦（同时刻≥2键）:", chords)
