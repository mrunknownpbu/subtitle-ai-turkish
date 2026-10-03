"""Logging setup: console + rotating file."""
import logging
import logging.handlers
import os
from pathlib import Path

FMT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"


def setup_logging(verbose: bool = False, log_dir: str | None = None) -> None:
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.DEBUG)

    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG if verbose else logging.INFO)
    console.setFormatter(logging.Formatter(FMT, "%H:%M:%S"))
    root.addHandler(console)

    log_dir = log_dir or os.environ.get("SUBAI_LOG_DIR", "/logs")
    try:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        fh = logging.handlers.RotatingFileHandler(
            Path(log_dir) / "subai.log", maxBytes=5_000_000, backupCount=3, encoding="utf-8"
        )
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter(FMT))
        root.addHandler(fh)
    except OSError as exc:
        logging.getLogger(__name__).warning("File logging disabled (%s): %s", log_dir, exc)

    for noisy in ("faster_whisper", "urllib3", "huggingface_hub", "filelock", "httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
