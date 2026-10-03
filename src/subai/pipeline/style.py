"""Natural spoken-tone punctuation, learned from the human Turkish subtitles of S01E01-05.

Two rules only, both validated on a held-out episode (S01E05) before being adopted:
  * continuation dots  - a sentence that runs across two cues ends the first with "..." and starts
    the second with "...", exactly as the human subtitlers do (precision 91%, recall 92%).
  * interjection "!"   - a sentence-final interjection (Aa, Hişt, Of, Bravo ...) takes "!"
    (precision 92%, recall 28%).
Comma insertion was tested and rejected (precision <= 52% at every threshold).
The interjection list lives in glossary/language/<lang>.yaml under `tone:`.
"""
import re

from subai.models import Cue

_W = "A-Za-z0-9_çğıöşüÇĞİÖŞÜ"
TERMINAL = (".", "?", "!", "…", "...")


def _exclaim(text: str, words: list[str]) -> str:
    for w in words:
        pat = re.escape(w)
        # "Aa." -> "Aa!"   (also mid-cue sentence ends)
        text = re.sub(rf"(?<![{_W}])({pat})\.(?=\s|$)", r"\1!", text, flags=re.IGNORECASE)
        # cue ends right after the interjection with no punctuation
        text = re.sub(rf"(?<![{_W}])({pat})$", r"\1!", text, flags=re.IGNORECASE)
    return text


def apply_tone(cues: list[Cue], interjections: list[str] | None = None,
               continuation_dots: bool = True, max_gap: float = 1.5) -> list[Cue]:
    out = [Cue(c.start, c.end, c.text) for c in cues]
    if interjections:
        for c in out:
            c.text = "\n".join(_exclaim(line, interjections) for line in c.text.split("\n"))
    if continuation_dots:
        for a, b in zip(out, out[1:]):
            if a.text.rstrip().endswith(TERMINAL) or b.start - a.end > max_gap:
                continue
            if a.text.lstrip().startswith("-") or b.text.lstrip().startswith("-"):
                continue  # dialogue turns are separate speakers, not one running sentence
            a.text = a.text.rstrip() + "..."
            if not b.text.lstrip().startswith(("...", "…")):
                b.text = "..." + b.text.lstrip()
    return out
