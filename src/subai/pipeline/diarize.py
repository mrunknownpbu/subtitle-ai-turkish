"""Speaker diarization with pyannote.audio 3.1 (who speaks when).

Run as a subprocess so PyTorch's CUDA libraries never share a process with faster-whisper's
(CTranslate2) and the GPU memory is fully released afterwards:

    python -m subai.pipeline.diarize AUDIO.wav OUT.json

OUT.json is a list of [start_seconds, end_seconds, "SPEAKER_00"]. The gated models need a
Hugging Face token (HF_TOKEN) once; after that they run from the local cache.
"""
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

log = logging.getLogger(__name__)
MODEL = "pyannote/speaker-diarization-3.1"


class DiarizationError(Exception):
    pass


def run_diarization(wav: Path, timeout: int = 3 * 3600) -> list[list]:
    """Diarize a 16 kHz mono WAV in a child process. Returns [[start, end, speaker], ...]."""
    out = wav.with_suffix(".diar.json")
    cmd = [sys.executable, "-m", "subai.pipeline.diarize", str(wav), str(out)]
    log.info("Running speaker diarization (%s)...", MODEL)
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        raise DiarizationError("diarization timed out") from exc
    if res.returncode != 0:
        tail = (res.stderr or res.stdout).strip().splitlines()[-3:]
        hint = ""
        if any(k in " ".join(tail) for k in ("401", "403", "gated", "token", "Cannot access")):
            hint = (" Accept the model terms at https://huggingface.co/pyannote/speaker-diarization-3.1 and "
                    "https://huggingface.co/pyannote/segmentation-3.0 and set HF_TOKEN.")
        if "No module named 'pyannote" in " ".join(tail):
            hint = " pyannote.audio is not installed (requirements/diarize.txt)."
        raise DiarizationError("diarization failed: " + " | ".join(tail) + hint)
    return json.loads(out.read_text(encoding="utf-8"))


def _main(wav: str, out: str) -> None:
    import torch
    from pyannote.audio import Pipeline

    token = os.environ.get("HF_TOKEN") or None
    try:
        pipe = Pipeline.from_pretrained(MODEL, use_auth_token=token)
    except TypeError:  # pyannote.audio >= 4 renamed the argument
        pipe = Pipeline.from_pretrained(MODEL, token=token)
    if pipe is None:
        raise SystemExit("Cannot access " + MODEL + " (gated: accept the terms and set HF_TOKEN)")
    if torch.cuda.is_available():
        pipe.to(torch.device("cuda"))
    diar = pipe(wav)
    segs = [[round(t.start, 3), round(t.end, 3), spk] for t, _, spk in diar.itertracks(yield_label=True)]
    Path(out).write_text(json.dumps(segs), encoding="utf-8")
    print(f"{len(segs)} segments, {len({s[2] for s in segs})} speakers")


if __name__ == "__main__":
    _main(sys.argv[1], sys.argv[2])
