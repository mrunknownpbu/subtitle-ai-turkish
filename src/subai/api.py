"""Local review API: list episodes, read Turkish/English cues, save an edited English cue. Serves web/dist."""
import os
import shutil
from pathlib import Path

import pysrt
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

ROOT = Path(os.environ.get("SUBAI_OUTPUT", "/output")).resolve()
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


if WEB.is_dir():  # last, so it never shadows /api
    app.mount("/", StaticFiles(directory=WEB, html=True), name="web")
