"""Local review API: list episodes, read Turkish/English cues, save an edited English cue. Serves web/dist."""
import json
import os
import re
import shutil
import subprocess
import sys
import queue
import threading
import time
import uuid
from functools import lru_cache
from pathlib import Path

import numpy as np
import pysrt
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from subai.glossary import detect_series_id
from subai.output import is_protected, write_srt_atomic, write_subs_atomic
from subai.pipeline.srtio import read_srt

ROOT = Path(os.environ.get("SUBAI_OUTPUT", "/output")).resolve()
MEDIA = Path(os.environ.get("SUBAI_MEDIA", "/data"))
RATE, PEAKS_PER_S = 4000, 20
MAX_SRT = 5_000_000
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
    if any(j["ep"] == ep and j["state"] in ("queued", "running") for j in RETIMES.values()):
        raise HTTPException(409, "a job is working on this episode; wait for it to finish before editing")
    subs = pysrt.open(str(en), encoding="utf-8")
    if not 0 <= i < len(subs):
        raise HTTPException(404, "cue not found")
    orig = en.with_name(en.name + ".orig")
    if not orig.exists():  # keep the machine output the first time a human edits it
        shutil.copy2(en, orig)
    subs[i].text = edit.en
    write_subs_atomic(en, subs, allow_overwrite=True)
    return {"i": i, "en": edit.en, "media_error": _mirror_to_media(ep, subs)}


def _mirror_to_media(ep: str, subs: pysrt.SubRipFile) -> str | None:
    """Also write the edited English subtitle as <video name>.en.srt beside the video, so the player library sees manual edits.
    The edit is already safe in /output, so a failure here is reported, never raised."""
    try:
        video, _ = _source(ep)
        if video is None:
            return "no video chosen for this episode"
        write_subs_atomic(video.with_name(video.stem + ".en.srt"), subs, allow_overwrite=True)
    except Exception as exc:  # read-only mount, permissions, protected name
        return str(exc)
    return None


VIDEO_EXT = {".mkv", ".mp4"}


@lru_cache(maxsize=None)
def _index() -> dict[str, Path]:
    """File name (no extension) -> video path for the whole library; one walk, ~10 s for 76k files."""
    return {p.stem: p for p in MEDIA.rglob("*") if p.suffix.lower() in VIDEO_EXT}


@app.on_event("startup")
def _warm_index() -> None:
    _load_jobs()
    threading.Thread(target=_index, daemon=True).start()


def _media(name: str) -> Path | None:
    p = _index().get(name)
    if p is None:
        _index.cache_clear()  # library may have changed; don't remember a miss
    return p


def _under(base: Path, rel: str) -> Path:
    p = (base / rel).resolve()
    if not p.is_relative_to(base.resolve()):
        raise HTTPException(400, "path outside the library")
    return p


def _source(ep: str) -> tuple[Path | None, bool]:
    """The video for an episode: the user's choice (<ep>.video) if set, else a file named like the episode. -> (path, auto)"""
    choice = (ROOT / f"{ep}.video")
    if choice.is_file():
        p = _under(MEDIA, choice.read_text().strip())
        if p.is_file():
            return p, False
    if ep.startswith("uploads/"):  # uploaded subtitles have no matching file: the user picks the video
        return None, True
    return _media(Path(ep).name), True


def _video(ep: str) -> Path:
    _pair(ep)
    p, _ = _source(ep)
    if p is None:
        raise HTTPException(404, "no video found for this episode")
    return p


class VideoChoice(BaseModel):
    file: str = ""  # path under the media library; "" = automatic match


@app.get("/api/browse")
def browse(path: str = "", kind: str = "video") -> dict:
    d = _under(MEDIA, path)
    if not d.is_dir():
        raise HTTPException(404, "folder not found")
    exts = {".srt"} if kind == "srt" else VIDEO_EXT
    kids = sorted((p for p in d.iterdir() if not p.name.startswith(".")), key=lambda p: p.name.lower())
    return {"path": path, "dirs": [p.name for p in kids if p.is_dir()],
            "files": [p.name for p in kids if p.is_file() and p.suffix.lower() in exts]}


@app.get("/api/video-choice")
def video_choice(ep: str) -> dict:
    _pair(ep)
    p, auto = _source(ep)
    return {"file": str(p.relative_to(MEDIA)) if p else "", "auto": auto}


@app.put("/api/video-choice")
def set_video_choice(ep: str, c: VideoChoice) -> dict:
    _pair(ep)
    if c.file:
        p = _under(MEDIA, c.file)
        if not p.is_file() or p.suffix.lower() not in VIDEO_EXT:
            raise HTTPException(400, "not a video file")
        (ROOT / f"{ep}.video").write_text(c.file)
    else:
        (ROOT / f"{ep}.video").unlink(missing_ok=True)
    for s in (".web.mp4", ".peaks.json"):  # derived from the old video
        (ROOT / f"{ep}{s}").unlink(missing_ok=True)
    return video_choice(ep)


def _store_srt(name: str, lang: str, data: bytes, replace: bool = False) -> str:
    """Save an uploaded SRT as uploads/<name>.<lang>.srt (an existing file is kept unless `replace`);
    the other language gets a blank copy with the same timings."""
    if lang not in ("tr", "en"):
        raise HTTPException(400, "lang must be tr or en")
    stem = _safe_stem(name)
    subs = _parse_srt(data)
    d = ROOT / "uploads"
    d.mkdir(exist_ok=True)
    dest = d / f"{stem}.{lang}.srt"
    # the blank copy made for the missing language is a placeholder, not the user's work: a real upload replaces it
    placeholder = dest.exists() and not any(c.text.strip() for c in pysrt.open(str(dest), encoding="utf-8"))
    if not write_subs_atomic(dest, subs, allow_overwrite=replace or placeholder):
        raise HTTPException(409, f"{stem}.{lang}.srt already exists: tick Replace to overwrite it, or choose another name")
    _ensure_pair(stem, lang)
    return f"uploads/{stem}"


def _safe_stem(name: str) -> str:
    stem = re.sub(r"[^\w .()\-]", "_", Path(name).stem).strip(" .")[:100]
    if not stem:
        raise HTTPException(400, "name required")
    return stem


def _parse_srt(data: bytes) -> pysrt.SubRipFile:
    subs = read_srt(data)
    if not len(subs):
        raise HTTPException(400, "no subtitle cues found in this file")
    return subs


def _ensure_pair(stem: str, lang: str) -> None:
    """uploads/<stem>.<lang>.srt exists; give the other language a blank copy with the same timings if it has none."""
    d = ROOT / "uploads"
    other = d / f"{stem}.{'en' if lang == 'tr' else 'tr'}.srt"
    if not other.exists():
        subs = pysrt.open(str(d / f"{stem}.{lang}.srt"), encoding="utf-8")
        for s in subs:
            s.text = ""
        write_subs_atomic(other, subs, allow_overwrite=False)


@app.post("/api/upload")
async def upload(request: Request, name: str, lang: str, replace: bool = False) -> dict:
    data = await request.body()
    if len(data) > MAX_SRT:
        raise HTTPException(413, "file too large for a subtitle")
    return {"ep": _store_srt(name, lang, data, replace)}


RETIMES: dict[str, dict] = {}
QUEUE: "queue.Queue[tuple]" = queue.Queue()  # jobs wait here; one worker runs them one at a time (the GPU is shared)
_worker_lock = threading.Lock()
_worker: threading.Thread | None = None


PROCS: dict[str, subprocess.Popen] = {}
MAX_JOBS = 50  # history kept on disk and listed


def _save_job(jid: str) -> None:
    """Persist a job as <output>/.jobs/<id>.json so history survives a restart; drop the oldest beyond MAX_JOBS."""
    d = ROOT / ".jobs"
    try:
        d.mkdir(exist_ok=True)
        (d / f"{jid}.json").write_text(json.dumps(RETIMES[jid]))
        for old in sorted(d.glob("*.json"), key=lambda f: f.stat().st_mtime)[:-MAX_JOBS]:
            RETIMES.pop(old.stem, None)
            old.unlink(missing_ok=True)
    except OSError:
        pass  # history is a convenience: never fail a job over it


def _load_jobs() -> None:
    """A job still 'running' on disk belonged to a process that died with the server."""
    for f in sorted((ROOT / ".jobs").glob("*.json")) if (ROOT / ".jobs").is_dir() else []:
        try:
            job = json.loads(f.read_text())
        except (OSError, ValueError):
            continue
        if job.get("state") in ("running", "queued"):
            job.update(state="failed", ended=job.get("ended") or time.time(), log=(job.get("log", []) + ["interrupted by a server restart"])[-40:])
            f.write_text(json.dumps(job))
        RETIMES[f.stem] = job


def _run_job(job: dict, steps: list[tuple], finish, cleanup=lambda: None, jid: str = "", prepare=lambda: None) -> None:
    """Run `subai` CLI steps one after another (stop at the first failure), streaming their log into the job.
    Exit code 2 from `retime` means refused: not enough evidence, nothing written."""
    try:
        if msg := prepare():
            job["log"] = (job["log"] + [msg])[-40:]
        for label, cmd, *keep in steps:
            if keep and keep[0] is not None and keep[0].exists():  # KEEP: never replace a file the user did not ask to replace
                job["log"] = (job["log"] + [f"== {label}", f"KEEP: {keep[0].name} already exists"])[-40:]
                continue
            job["log"] = (job["log"] + [f"== {label}"])[-40:]
            job["step"] = label
            proc = PROCS[jid] = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for line in proc.stdout:  # type: ignore[union-attr]
                job["log"] = (job["log"] + [line.rstrip().split(" ", 2)[-1]])[-40:]
            rc = proc.wait()
            if rc:
                job["state"] = "cancelled" if job.get("cancelled") else "refused" if rc == 2 and cmd[3] == "retime" else "failed"
                return
        finish()
        job["state"] = "done"
    except Exception as exc:  # report, never leave the job "running" forever
        job["log"].append(str(exc))
        job["state"] = "failed"
    finally:
        PROCS.pop(jid, None)
        job["ended"] = time.time()
        _save_job(jid)
        cleanup()


def _work() -> None:
    while True:
        jid, steps, done, cleanup, prepare = QUEUE.get()
        job = RETIMES.get(jid)
        if job is None or job["state"] != "queued":  # cancelled while waiting
            cleanup()
            continue
        job.update(state="running", began=time.time())
        _save_job(jid)
        _run_job(job, steps, done, cleanup, jid, prepare)


def _publish(jid: str, ep: str, video: str) -> None:
    """Copy the Turkish/English subtitles this job wrote to <video name>.tr/.en.srt beside the video. Files a step kept
    (older than the job), blank placeholders and <name>.retimed variants are not copied. A failure is logged, never fatal."""
    job, v = RETIMES[jid], MEDIA / video
    if Path(ep).name != v.stem:
        return
    for lang in ("tr", "en"):
        f = ROOT / f"{ep}.{lang}.srt"
        try:
            if not f.is_file() or f.stat().st_mtime < job["began"] or not any(c.text.strip() for c in pysrt.open(str(f), encoding="utf-8")):
                continue
            write_srt_atomic(v.with_name(f"{v.stem}.{lang}.srt"), f.read_text(encoding="utf-8"), allow_overwrite=True)
            job["log"] = (job["log"] + [f"Copied {lang}.srt to the media folder"])[-40:]
        except Exception as exc:  # read-only mount, permissions, protected name
            job["log"] = (job["log"] + [f"Not copied to the media folder ({lang}): {exc}"])[-40:]


def _start_job(steps: list[tuple], ep: str, video: str, finish=lambda: None, cleanup=lambda: None,
               title: str = "", prepare=lambda: None) -> dict:
    """Queue a job; the single worker runs jobs in order. `prepare` runs just before its steps. Links the episode to its video when done."""
    global _worker
    job = {"state": "queued", "log": [], "ep": ep, "title": title or " + ".join(st[0] for st in steps),
           "video": Path(video).name, "started": time.time(), "began": None, "ended": None, "step": ""}
    jid = uuid.uuid4().hex
    RETIMES[jid] = job
    _save_job(jid)

    def done() -> None:
        finish()
        (ROOT / f"{ep}.video").write_text(video)
        _publish(jid, ep, video)

    QUEUE.put((jid, steps, done, cleanup, prepare))
    with _worker_lock:
        if _worker is None:
            _worker = threading.Thread(target=_work, daemon=True)
            _worker.start()
    return {"id": jid}


@app.post("/api/retime-job-cancel")
def cancel_job(id: str) -> dict:
    job = RETIMES.get(id)
    if job is not None and job["state"] == "queued":
        job.update(state="cancelled", ended=time.time())  # the worker skips it and cleans up
        _save_job(id)
        return {"ok": True}
    proc = PROCS.get(id)
    if proc is None:
        raise HTTPException(404, "no running or queued job with this id")
    job["cancelled"] = True  # type: ignore[index]
    proc.terminate()  # the job thread then sees a non-zero exit and marks it cancelled
    return {"ok": True}


@app.get("/api/jobs")
def jobs() -> list[dict]:
    """Job history, newest first (running jobs included)."""
    return [{"id": i, **j} for i, j in sorted(RETIMES.items(), key=lambda kv: -kv[1].get("started", 0))]


@app.post("/api/jobs-clear")
def jobs_clear() -> dict:
    """Forget every finished job."""
    gone = [i for i, j in RETIMES.items() if j["state"] != "running"]
    for i in gone:
        RETIMES.pop(i)
        (ROOT / ".jobs" / f"{i}.json").unlink(missing_ok=True)
    return {"cleared": len(gone)}


def _read_source(src: str) -> bytes:
    """A Turkish SRT chosen from the media library. Reference subtitles (.en.hi.srt ...) are never read as input."""
    s = _under(MEDIA, src)
    if is_protected(s):
        raise HTTPException(400, "that is a protected reference subtitle; it is never used as input")
    if not s.is_file() or s.suffix.lower() != ".srt":
        raise HTTPException(400, "src is not an .srt file")
    return s.read_bytes()


def _retimed_stem(stem: str, replace: bool) -> str:
    """Re-timing keeps an existing uploads/<stem>.tr.srt and writes <stem>.retimed beside it unless `replace`."""
    return stem if replace or not (ROOT / "uploads" / f"{stem}.tr.srt").exists() else f"{stem}.retimed"


@app.post("/api/retime")
async def retime(request: Request, video: str, name: str, src: str = "", replace_original: bool = False) -> dict:
    """Re-time a Turkish SRT (request body, or `src` under the media library) onto `video`'s audio and open it as
    the new episode uploads/<name>. An existing episode of that name is kept: the result is saved as <name>.retimed
    unless `replace_original`. Queued background job: poll /api/retime-job."""
    v = _under(MEDIA, video)
    if not v.is_file() or v.suffix.lower() not in VIDEO_EXT:
        raise HTTPException(400, "not a video file")
    stem = _safe_stem(name)
    data = await request.body()
    if not data:
        if not src:
            raise HTTPException(400, "send an SRT file or choose one from the library")
        data = _read_source(src)
    if len(data) > MAX_SRT:
        raise HTTPException(413, "file too large for a subtitle")
    subs = _parse_srt(data)
    stem = _retimed_stem(stem, replace_original)
    scratch = ROOT / "uploads" / f".{stem}.{uuid.uuid4().hex[:8]}.incoming.srt"  # unique: the same name may be queued twice
    try:
        scratch.parent.mkdir(exist_ok=True)
        write_subs_atomic(scratch, subs, allow_overwrite=True)
        dest = ROOT / "uploads" / f"{stem}.tr.srt"
        cmd = [sys.executable, "-m", "subai", "retime", "-i", str(scratch), "--video", str(v),
               "-o", str(dest), "--search-root", str(ROOT), "--retranscribe"]
        return _start_job([("Re-timing", cmd, None if replace_original else dest)], f"uploads/{stem}", str(v.relative_to(MEDIA)),
                          finish=lambda: _ensure_pair(stem, "tr"), cleanup=lambda: scratch.unlink(missing_ok=True))
    except Exception:
        scratch.unlink(missing_ok=True)
        raise


FT = Path(os.environ.get("SUBAI_FT", "/ft"))


@app.get("/api/models")
def models() -> dict:
    """Whisper models offered for transcription: fine-tuned ones in /ft first, then the stock large-v3."""
    tuned = sorted(str(p) for p in FT.glob("*") if p.is_dir()) if FT.is_dir() else []
    return {"models": tuned[::-1] + [os.environ.get("SUBAI_MODEL", "large-v3")]}


@app.post("/api/process")
async def process(request: Request, video: str, transcribe: bool = True, translate: bool = True,
                  src: str = "", model: str = "", retime: bool = False,
                  overwrite_original: bool = False, overwrite_english: bool = False) -> dict:
    """Transcribe a library video to Turkish, re-time a supplied Turkish SRT onto its audio, and/or translate to English,
    as the episode uploads/<video name>. Steps run in that order. Re-timing needs an SRT (request body or `src`) and
    replaces transcription as the source of the Turkish text. Translate-only needs an SRT or the episode's earlier one.
    Existing files are kept (the step is skipped, logged as KEEP) unless `overwrite_original` (Turkish) or
    `overwrite_english`; re-timing writes <name>.retimed beside an existing Turkish file instead."""
    v = _under(MEDIA, video)
    if not v.is_file() or v.suffix.lower() not in VIDEO_EXT:
        raise HTTPException(400, "not a video file")
    if not (transcribe or translate or retime):
        raise HTTPException(400, "choose at least one step")
    if retime and transcribe:
        raise HTTPException(400, "re-timing moves an existing subtitle; a fresh transcription is already timed to the audio")
    stem = _retimed_stem(v.stem, overwrite_original) if retime else v.stem
    tr, en = ROOT / "uploads" / f"{stem}.tr.srt", ROOT / "uploads" / f"{stem}.en.srt"
    if model and model != os.environ.get("SUBAI_MODEL", "large-v3") and not _under(FT, model.removeprefix(str(FT) + "/")).is_dir():
        raise HTTPException(400, "unknown model")
    data = await request.body()
    if not data and src:
        data = _read_source(src)
    if len(data) > MAX_SRT:
        raise HTTPException(413, "file too large for a subtitle")
    if retime and not data:
        raise HTTPException(400, "re-timing needs a Turkish SRT: upload one or choose it from the library")
    if translate and not transcribe and not data and not tr.is_file():
        raise HTTPException(400, "translating needs a Turkish SRT: upload one or choose it from the library")
    parsed = _parse_srt(data) if data else None
    scratch = ROOT / "uploads" / f".{stem}.{uuid.uuid4().hex[:8]}.incoming.srt"
    try:
        (ROOT / "uploads").mkdir(exist_ok=True)
        py = [sys.executable, "-m", "subai"]
        steps = []
        if retime:
            write_subs_atomic(scratch, parsed, allow_overwrite=True)  # type: ignore[arg-type]
            steps.append(("Re-timing", py + ["retime", "-i", str(scratch), "--video", str(v), "-o", str(tr),
                                             "--search-root", str(ROOT), "--retranscribe"], None if overwrite_original else tr))
        if transcribe:
            steps.append(("Transcribing audio", py + ["run", "-i", str(v), "-o", str(ROOT / "uploads"), "--model",
                                                      model or os.environ.get("SUBAI_MODEL", "large-v3"), "--overwrite", "--retranscribe"],
                          None if overwrite_original else tr))
        if translate:
            sid = detect_series_id(v)
            steps.append(("Translating to English", py + ["translate", "-i", str(tr), "-o", str(en)] + (["--series", sid] if sid else []),
                          None if overwrite_english else en))
        # a supplied SRT used as is is written when the job starts, so a queued job never overwrites a running one's input
        def use_as_is() -> str | None:
            if parsed is None or retime or transcribe:
                return None
            if write_subs_atomic(tr, parsed, allow_overwrite=overwrite_original):
                return None
            return f"KEEP: {tr.name} already exists, so the Turkish subtitle you sent was not used (tick Replace to use it)"

        return _start_job(steps, f"uploads/{stem}", str(v.relative_to(MEDIA)), finish=lambda: _ensure_pair(stem, "tr"),
                          cleanup=lambda: scratch.unlink(missing_ok=True), prepare=use_as_is)
    except Exception:
        scratch.unlink(missing_ok=True)
        raise


@app.get("/api/retime-job")
def retime_job(id: str) -> dict:
    job = RETIMES.get(id)
    if job is None:
        raise HTTPException(404, "unknown job")
    return job


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
