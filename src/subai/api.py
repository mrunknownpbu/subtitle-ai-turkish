"""Local review API: list episodes, read Turkish/English cues, save an edited English cue. Serves web/dist."""
import json
import os
import shutil
import subprocess
import threading
from functools import lru_cache
from pathlib import Path

import numpy as np
import pysrt
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

ROOT = Path(os.environ.get("SUBAI_OUTPUT", "/output")).resolve()
MEDIA = Path(os.environ.get("SUBAI_MEDIA", "/data"))
RATE, PEAKS_PER_S = 4000, 20
WEB = Path(os.environ.get("SUBAI_WEB", "/web"))

app = FastAPI(title="subai review")


class CueEdit(BaseModel):
    en: str


def _pair(ep: str) -> tuple[Path, Path]:
    tr = (ROOT / f"{ep}.tr.srt").resolve()
    if not tr.is_relative_to(ROOT) or not tr.is_file():
        raise HTTPException(404, "episode not found")
    en = tr.with_name(tr.name.removesuffix(".tr.srt") + ".en.srt")
    if not en.is_file():
        raise HTTPException(404, "episode has no English subtitle")
    return tr, en


@app.get("/api/episodes")
def episodes() -> list[str]:
    return sorted(str(p.relative_to(ROOT)).removesuffix(".tr.srt") for p in ROOT.rglob("*.tr.srt")
                  if p.with_name(p.name.removesuffix(".tr.srt") + ".en.srt").is_file())


@app.get("/api/cues")
def cues(ep: str) -> list[dict]:
    tr, en = _pair(ep)
    a, b = pysrt.open(str(tr), encoding="utf-8"), pysrt.open(str(en), encoding="utf-8")
    return [{"i": i, "start": s.start.ordinal / 1000, "end": s.end.ordinal / 1000,
             "tr": s.text, "en": b[i].text if i < len(b) else ""} for i, s in enumerate(a)]


@app.put("/api/cues/{i}")
def save_cue(i: int, ep: str, edit: CueEdit) -> dict:
    _, en = _pair(ep)
    subs = pysrt.open(str(en), encoding="utf-8")
    if not 0 <= i < len(subs):
        raise HTTPException(404, "cue not found")
    orig = en.with_name(en.name + ".orig")
    if not orig.exists():  # keep the machine output the first time a human edits it
        shutil.copy2(en, orig)
    subs[i].text = edit.en
    tmp = en.with_name(en.name + ".part")
    subs.save(str(tmp), encoding="utf-8")
    tmp.replace(en)
    return {"i": i, "en": edit.en}


@lru_cache(maxsize=None)
def _media(name: str) -> Path | None:
    for ext in ("mkv", "mp4"):
        for p in MEDIA.rglob(f"{name}.{ext}"):
            return p
    return None


def _video(ep: str) -> Path:
    _pair(ep)
    p = _media(Path(ep).name)
    if p is None:
        _media.cache_clear()  # don't remember a miss
        raise HTTPException(404, "no video found for this episode")
    return p


_remux_lock = threading.Lock()


@app.get("/api/video")
def video(ep: str) -> FileResponse:  # FileResponse handles Range requests, so seeking works
    """Browsers (Firefox) can't play MKV: stream-copy to MP4 once (no re-encode), cache it, serve that."""
    src = _video(ep)
    if src.suffix == ".mp4":
        return FileResponse(src, media_type="video/mp4")
    mp4 = (ROOT / f"{ep}.web.mp4").resolve()
    with _remux_lock:  # ponytail: one remux at a time, a second tab waits
        if not mp4.is_file():
            tmp = mp4.with_name(mp4.name + ".part")
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src), "-map", "0:v:0", "-map", "0:a:0?", "-c", "copy",
                            "-movflags", "+faststart", "-f", "mp4", str(tmp)], check=True)
            tmp.replace(mp4)
    return FileResponse(mp4, media_type="video/mp4")


@app.get("/api/peaks")
def peaks(ep: str) -> dict:
    """Waveform: max |sample| (0-255) per 1/PEAKS_PER_S second, decoded once and cached next to the subtitles."""
    cache = (ROOT / f"{ep}.peaks.json").resolve()
    if cache.is_file():
        return json.loads(cache.read_text())
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(_video(ep)), "-vn", "-ac", "1", "-ar", str(RATE),
                          "-f", "s16le", "-"], capture_output=True, check=True).stdout
    a = np.abs(np.frombuffer(raw, dtype=np.int16).astype(np.int32))
    step = RATE // PEAKS_PER_S
    a = a[: len(a) // step * step].reshape(-1, step).max(axis=1)
    out = {"rate": PEAKS_PER_S, "peaks": (a * 255 // max(int(a.max()), 1)).tolist()}
    cache.write_text(json.dumps(out))
    return out


if WEB.is_dir():  # last, so it never shadows /api
    app.mount("/", StaticFiles(directory=WEB, html=True), name="web")
