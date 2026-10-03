"""Single-file and batch processing."""
import json
import logging
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from subai.glossary import apply_corrections, is_hallucination
from subai.models import Word
from subai.pipeline.dialogue import assign_speakers, merge_dialogue
from subai.pipeline.diarize import MODEL as DIAR_MODEL
from subai.pipeline.diarize import DiarizationError, run_diarization
from subai.pipeline.style import apply_tone
from subai.pipeline.audio import MEDIA_EXT, MediaError, extract_audio
from subai.pipeline.segmenter import build_cues
from subai.pipeline.srtio import write_srt
from subai.pipeline.transcribe import Transcriber, TranscriptionError

log = logging.getLogger(__name__)


@dataclass
class Result:
    input: str
    status: str  # ok | skipped | failed
    output: str = ""
    seconds: float = 0.0
    cues: int = 0
    error: str = ""


def discover(inputs: list[Path], input_dir: Path | None, recursive: bool) -> list[tuple[Path, Path]]:
    """Return (media_file, relative_parent) pairs. Skips non-media and hidden files."""
    found: list[tuple[Path, Path]] = []
    for p in inputs:
        if p.is_file():
            found.append((p, Path(".")))
        else:
            log.error("Input not found or not a file: %s", p)
    if input_dir is not None:
        if not input_dir.is_dir():
            log.error("Input directory not found: %s", input_dir)
        else:
            it = input_dir.rglob("*") if recursive else input_dir.glob("*")
            for f in sorted(it):
                if f.is_file() and f.suffix.lower() in MEDIA_EXT and not f.name.startswith("."):
                    found.append((f, f.parent.relative_to(input_dir)))
    return found


def _cache_key(src: Path, tr: Transcriber, track: int | None) -> dict:
    st = src.stat()
    return {"v": 1, "size": st.st_size, "mtime": int(st.st_mtime), "track": track, **tr.settings()}


def _diarization(src: Path, out_path: Path, track: int | None) -> list[list] | None:
    """Cached speaker segments, or None if diarization is unavailable (subtitles still get written)."""
    cache = out_path.parent / ".subai" / f"{src.stem}.diar.json"
    st = src.stat()
    key = {"v": 1, "size": st.st_size, "mtime": int(st.st_mtime), "track": track, "model": DIAR_MODEL}
    if cache.exists():
        try:
            data = json.loads(cache.read_text(encoding="utf-8"))
            if data.get("key") == key:
                log.info("Using cached diarization (%d segments)", len(data["segments"]))
                return data["segments"]
        except (OSError, ValueError, KeyError, TypeError):
            log.warning("Ignoring unreadable cache %s", cache)
    try:
        with tempfile.TemporaryDirectory(prefix="subai_diar_") as tmp:
            wav = Path(tmp) / "audio.wav"
            extract_audio(src, wav, track)
            segments = run_diarization(wav)
    except (DiarizationError, MediaError, OSError) as exc:
        log.warning("Diarization skipped: %s", exc)
        return None
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({"key": key, "segments": segments}), encoding="utf-8")
    return segments


def process_file(src: Path, out_path: Path, tr: Transcriber, track: int | None,
                 corrections: list[dict] | None = None, force_asr: bool = False,
                 hallucinations: list[str] | None = None, tone: dict | None = None,
                 diarize: bool = False) -> Result:
    """ASR words are cached next to the output (.subai/<name>.words.json), so changes to
    segmentation, timing or glossary corrections re-run in seconds without re-transcribing."""
    t0 = time.time()
    cache = out_path.parent / ".subai" / f"{src.stem}.words.json"
    key = _cache_key(src, tr, track)
    words: list[Word] | None = None
    duration = 0.0
    if cache.exists() and not force_asr:
        try:
            data = json.loads(cache.read_text(encoding="utf-8"))
            if data.get("key") == key:
                words = [Word(*w) for w in data["words"]]
                duration = float(data["duration"])
                log.info("Using cached transcription (%d words); --retranscribe to redo", len(words))
        except (OSError, ValueError, KeyError, TypeError):
            log.warning("Ignoring unreadable cache %s", cache)
    if words is None:
        with tempfile.TemporaryDirectory(prefix="subai_") as tmp:
            wav = Path(tmp) / "audio.wav"
            duration = extract_audio(src, wav, track)
            log.info("Audio extracted (%.1f min)", duration / 60)
            words = tr.transcribe(wav, duration)
        if words:
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps({"key": key, "duration": duration,
                                         "words": [[w.text, w.start, w.end] for w in words]},
                                        ensure_ascii=False), encoding="utf-8")
    if not words:
        raise TranscriptionError("no speech detected")
    if diarize:
        segments = _diarization(src, out_path, track)
        if segments:
            words = assign_speakers(words, segments)
            log.info("Speakers assigned (%d distinct)", len({w.speaker for w in words if w.speaker}))
    cues = build_cues(words)
    if hallucinations:
        kept = [c for c in cues if not is_hallucination(c.text, hallucinations)]
        if len(kept) != len(cues):
            log.info("Dropped %d hallucinated cue(s) (e.g. %r)", len(cues) - len(kept),
                     next(c.text for c in cues if is_hallucination(c.text, hallucinations)))
        cues = kept
    if diarize:
        cues = merge_dialogue(cues)
    fixed = 0
    if corrections:
        for c in cues:
            c.text, k = apply_corrections(c.text, corrections)
            fixed += k
        if fixed:
            log.info("Applied %d glossary name correction(s)", fixed)
    if tone:
        cues = apply_tone(cues, tone.get("exclamation_interjections"), tone.get("continuation_dots", True),
                          float(tone.get("continuation_max_gap", 1.5)))
    write_srt(cues, out_path)
    secs = time.time() - t0
    rtf = duration / secs if secs else 0
    log.info("Wrote %s (%d cues, %.0fs, %.1fx realtime)", out_path, len(cues), secs, rtf)
    return Result(str(src), "ok", str(out_path), round(secs, 1), len(cues))


def run_batch(files: list[tuple[Path, Path]], out_dir: Path, tr: Transcriber,
              lang: str, overwrite: bool, track: int | None,
              corrections: list[dict] | None = None, force_asr: bool = False,
              hallucinations: list[str] | None = None, tone: dict | None = None,
              diarize: bool = False) -> list[Result]:
    results: list[Result] = []
    try:
        for n, (src, rel) in enumerate(files, 1):
            out_path = out_dir / rel / f"{src.stem}.{lang}.srt"
            log.info("[%d/%d] %s", n, len(files), src.name)
            if out_path.exists() and not overwrite:
                log.info("  skipped (exists; use --overwrite)")
                results.append(Result(str(src), "skipped", str(out_path)))
                continue
            try:
                results.append(process_file(src, out_path, tr, track, corrections, force_asr, hallucinations, tone, diarize))
            except (MediaError, TranscriptionError, OSError) as exc:
                log.error("  FAILED: %s", exc)
                results.append(Result(str(src), "failed", error=str(exc)))
            except Exception as exc:  # keep the batch going
                log.exception("  FAILED (unexpected): %s", exc)
                results.append(Result(str(src), "failed", error=f"unexpected: {exc}"))
    finally:
        tr.close()
        if results:
            try:
                out_dir.mkdir(parents=True, exist_ok=True)
                (out_dir / "batch_report.json").write_text(
                    json.dumps([asdict(r) for r in results], indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
            except OSError as exc:
                log.warning("Could not write batch report: %s", exc)
    return results
