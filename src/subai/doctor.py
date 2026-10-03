"""Environment self-check."""
import shutil
import subprocess
import tempfile
from pathlib import Path


def _check(name: str, ok: bool, detail: str = "") -> bool:
    print(f"[{'OK' if ok else 'FAIL'}] {name}{': ' + detail if detail else ''}")
    return ok


def run_doctor(model: str, out_dir: Path) -> bool:
    all_ok = True
    for tool in ("ffmpeg", "ffprobe"):
        all_ok &= _check(tool, shutil.which(tool) is not None)

    try:
        import ctranslate2

        n = ctranslate2.get_cuda_device_count()
        _check("GPU (CUDA)", n > 0, f"{n} device(s)" if n else "none found; CPU mode will be very slow")
    except Exception as exc:
        all_ok &= _check("ctranslate2", False, str(exc))

    free_gb = shutil.disk_usage(out_dir if out_dir.exists() else Path("/")).free / 1e9
    all_ok &= _check("disk space", free_gb > 10, f"{free_gb:.0f} GB free")

    try:
        from faster_whisper import WhisperModel  # noqa: F401
        from faster_whisper.utils import download_model

        path = download_model(model, local_files_only=True)
        all_ok &= _check(f"model '{model}' cached", True, path)
    except Exception:
        all_ok &= _check(f"model '{model}' cached", False, "run `subai download-model` first")
        return all_ok

    try:
        from subai.pipeline.transcribe import Transcriber

        with tempfile.TemporaryDirectory() as tmp:
            wav = Path(tmp) / "silence.wav"
            subprocess.run(
                ["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono",
                 "-t", "3", str(wav)], check=True,
            )
            t = Transcriber(model=model)
            t.transcribe(wav, 3)
            t.close()
        all_ok &= _check("smoke test (3 s of silence)", True)
    except Exception as exc:
        all_ok &= _check("smoke test", False, str(exc)[:200])
    return all_ok
