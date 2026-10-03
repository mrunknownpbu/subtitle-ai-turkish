"""faster-whisper transcription with automatic GPU -> smaller GPU -> CPU fallback."""
import gc
import logging
import time
from pathlib import Path

from subai.models import Word

log = logging.getLogger(__name__)

FALLBACKS = [("cuda", "int8_float16"), ("cuda", "int8"), ("cpu", "int8")]

# More sensitive than the library default (threshold 0.5): recovers short utterances such as
# "Hadi", "Evren" at VAD boundaries. Chosen from controlled comparisons on a Turkish drama clip.
VAD_PARAMS = {"threshold": 0.3, "min_silence_duration_ms": 300, "speech_pad_ms": 400}


class TranscriptionError(Exception):
    pass


def load_wav(path: Path):
    """Read 16 kHz mono PCM16 WAV (as written by audio.extract_audio) into float32 [-1, 1]."""
    import wave

    import numpy as np

    with wave.open(str(path), "rb") as wf:
        if wf.getframerate() != 16000 or wf.getnchannels() != 1 or wf.getsampwidth() != 2:
            raise TranscriptionError("expected 16 kHz mono 16-bit WAV")
        raw = wf.readframes(wf.getnframes())
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


MAX_WORD_GAP = 3.0  # seconds between words of one Whisper segment before we suspect a bad timestamp
MAX_STRAY = 3       # a misplaced group is at most this many words


def _shift(group: list[Word], delta: float) -> None:
    for w in group:
        w.start += delta
        w.end += delta


def repair_stray_words(ws: list[Word]) -> list[Word]:
    """Whisper sometimes time-stamps the first (or last) words of a segment far from the rest,
    e.g. 'Bu' 108 s before 'arada Serkanlar nerede?'. Move the short stray group next to the main one."""
    ws = [Word(w.text, w.start, w.end) for w in ws]
    for _ in range(3):
        for i in range(len(ws) - 1):
            if ws[i + 1].start - ws[i].end <= MAX_WORD_GAP:
                continue
            head, tail = ws[: i + 1], ws[i + 1:]
            if len(head) <= MAX_STRAY and len(head) < len(tail):
                _shift(head, ws[i + 1].start - 0.05 - head[-1].end)
            elif len(tail) <= MAX_STRAY and len(tail) < len(head):
                _shift(tail, ws[i].end + 0.05 - tail[0].start)
            else:
                continue
            break
        else:
            break
    return ws


def _cuda_available() -> bool:
    try:
        import ctranslate2

        return ctranslate2.get_cuda_device_count() > 0
    except Exception:
        return False


class Transcriber:
    def __init__(self, model: str = "large-v3", device: str = "auto",
                 compute_type: str = "auto", lang: str = "tr", beam_size: int = 5,
                 initial_prompt: str | None = None):
        self.model_name = model
        self.lang = lang
        self.beam_size = beam_size
        self.initial_prompt = initial_prompt
        if device == "auto" and compute_type == "auto":
            self.configs = FALLBACKS if _cuda_available() else [("cpu", "int8")]
        else:
            dev = "cuda" if device == "auto" and _cuda_available() else ("cpu" if device == "auto" else device)
            ct = compute_type if compute_type != "auto" else ("int8_float16" if dev == "cuda" else "int8")
            self.configs = [(dev, ct)]
        self._idx = 0
        self._model = None

    def _ensure(self):
        if self._model is None:
            from faster_whisper import WhisperModel

            device, ct = self.configs[self._idx]
            log.info("Loading model %s on %s (%s)", self.model_name, device, ct)
            t0 = time.time()
            try:
                self._model = WhisperModel(self.model_name, device=device, compute_type=ct)
            except Exception as exc:
                msg = str(exc).lower()
                if "out of memory" in msg or "cudnn" in msg or "cublas" in msg or "cuda" in msg:
                    raise
                raise TranscriptionError(
                    f"Could not load model '{self.model_name}': {exc}. "
                    "If this is the first run, the model must be downloaded "
                    "(`subai download-model`) with network access."
                ) from exc
            log.info("Model loaded in %.1fs", time.time() - t0)
        return self._model

    def _advance(self, reason: str) -> None:
        self._model = None
        gc.collect()
        if self._idx + 1 >= len(self.configs):
            raise TranscriptionError(f"No fallback left ({reason})")
        self._idx += 1
        log.warning("%s; falling back to %s/%s", reason, *self.configs[self._idx])

    def transcribe(self, wav: Path, duration: float = 0.0) -> list[Word]:
        audio = load_wav(wav)
        while True:
            try:
                model = self._ensure()
                segments, info = model.transcribe(
                    audio,
                    language=self.lang,
                    beam_size=self.beam_size,
                    vad_filter=True,
                    vad_parameters=VAD_PARAMS,
                    initial_prompt=self.initial_prompt,
                    condition_on_previous_text=False,
                    word_timestamps=True,
                    no_speech_threshold=0.6,
                    compression_ratio_threshold=2.4,
                    hallucination_silence_threshold=2.0,
                )
                words: list[Word] = []
                next_pct = 10
                for seg in segments:
                    words.extend(repair_stray_words(
                        [Word(w.word.strip(), w.start, w.end) for w in seg.words or []]))
                    if duration:
                        pct = int(seg.end / duration * 100)
                        if pct >= next_pct:
                            log.info("  transcribing... %d%%", min(pct, 100))
                            next_pct = (pct // 10 + 1) * 10
                return words
            except TranscriptionError:
                raise
            except Exception as exc:
                msg = str(exc).lower()
                if any(k in msg for k in ("out of memory", "cudnn", "cublas", "cuda")):
                    self._advance(f"GPU problem: {str(exc)[:120]}")
                    continue
                raise TranscriptionError(f"Transcription failed: {exc}") from exc

    def settings(self) -> dict:
        """Everything that changes the transcription result (used as the cache key)."""
        return {"model": self.model_name, "lang": self.lang, "beam": self.beam_size,
                "prompt": self.initial_prompt, "vad": VAD_PARAMS, "algo": 2}  # algo: bump when decoding/repair logic changes

    def close(self) -> None:
        self._model = None
        gc.collect()
