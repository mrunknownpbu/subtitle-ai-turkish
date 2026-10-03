"""Media probing and audio extraction via ffmpeg/ffprobe."""
import json
import logging
import subprocess
from pathlib import Path

log = logging.getLogger(__name__)

VIDEO_EXT = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".ts", ".m4v"}
AUDIO_EXT = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".opus", ".aac"}
MEDIA_EXT = VIDEO_EXT | AUDIO_EXT


class MediaError(Exception):
    """Unreadable, corrupt, or audio-less media."""


def probe(path: Path) -> dict:
    cmd = [
        "ffprobe", "-v", "error", "-print_format", "json",
        "-show_entries", "format=duration:stream=index,codec_type:stream_tags=language",
        str(path),
    ]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    except FileNotFoundError as exc:
        raise MediaError("ffprobe not found; ffmpeg is not installed") from exc
    except subprocess.TimeoutExpired as exc:
        raise MediaError("ffprobe timed out; file may be corrupt") from exc
    if out.returncode != 0:
        raise MediaError(f"ffprobe failed: {out.stderr.strip()[:300]}")
    return json.loads(out.stdout or "{}")


def pick_audio_stream(info: dict, track: int | None) -> int:
    """Return the 0-based index among audio streams. Prefers Turkish, else the first."""
    audio = [s for s in info.get("streams", []) if s.get("codec_type") == "audio"]
    if not audio:
        raise MediaError("no audio stream found")
    if track is not None:
        if not 0 <= track < len(audio):
            raise MediaError(f"audio track {track} not found (file has {len(audio)})")
        return track
    for i, s in enumerate(audio):
        if s.get("tags", {}).get("language", "").lower() in {"tur", "tr"}:
            return i
    return 0


def extract_audio(src: Path, dst: Path, track: int | None = None) -> float:
    """Write mono 16 kHz WAV to dst. Returns duration in seconds."""
    info = probe(src)
    duration = float(info.get("format", {}).get("duration") or 0)
    idx = pick_audio_stream(info, track)
    cmd = [
        "ffmpeg", "-v", "error", "-y", "-i", str(src),
        "-map", f"0:a:{idx}", "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(dst),
    ]
    log.debug("ffmpeg: %s", " ".join(cmd))
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise MediaError(f"ffmpeg failed: {res.stderr.strip()[:300]}")
    if not dst.exists() or dst.stat().st_size < 1000:
        raise MediaError("extracted audio is empty")
    return duration
