"""Turn word timestamps into readable subtitle cues (max 2 lines x 42 chars)."""
from subai.models import Cue, Word

MAX_LINE = 42
MAX_CHARS = 84
MAX_DUR = 6.5
MIN_DUR = 1.0
MAX_CPS = 17.0
MAX_EXTEND = 1.5  # seconds a cue may be extended past the last word
GAP_SPLIT = 0.3   # pause (s) that starts a new cue; tuned on S01E01-05 vs human subtitle breaks (F1 0.704 -> 0.729)
TURN_GAP = 0.25   # pause (s) that makes a mid-sentence speaker change believable
SENT_MIN = 30     # split after . ? ! once the cue has at least this many characters
COMMA_MIN = 60    # split after a comma once the cue has at least this many characters
MIN_GAP = 0.08
SENT_END = (".", "?", "!", "…")


def _wrap(text: str) -> str:
    if len(text) <= MAX_LINE:
        return text
    words = text.split(" ")
    best = None
    for i in range(1, len(words)):
        a, b = " ".join(words[:i]), " ".join(words[i:])
        score = abs(len(a) - len(b)) + (0 if a.endswith((",", ".", "?", "!", ":", ";")) else 3)
        if len(a) > MAX_LINE or len(b) > MAX_LINE:
            score += 100
        if best is None or score < best[0]:
            best = (score, a, b)
    return f"{best[1]}\n{best[2]}" if best else text


def _drop_repeats(cues: list[Cue]) -> list[Cue]:
    """Drop Whisper hallucination loops: same text 3+ times in a row."""
    out: list[Cue] = []
    run = 0
    for c in cues:
        if out and c.text == out[-1].text:
            run += 1
            if run >= 2:
                continue
        else:
            run = 0
        out.append(c)
    return out


def _fix_timing(cues: list[Cue]) -> list[Cue]:
    for i, c in enumerate(cues):
        c.start = max(0.0, c.start)
        if c.end <= c.start:
            c.end = c.start + 0.5
        nxt = cues[i + 1].start if i + 1 < len(cues) else None
        # Whisper word ends are tight: hold the text on screen long enough to read (MAX_CPS),
        # extending into the silence that follows, never into the next cue.
        want = max(MIN_DUR, len(c.text.replace("\n", " ")) / MAX_CPS)
        orig_end = c.end
        if c.end - c.start < want:
            target = min(c.start + want, orig_end + MAX_EXTEND)
            limit = nxt - MIN_GAP if nxt is not None else target
            c.end = max(orig_end, min(target, limit), c.start + 0.3)
        if nxt is not None and nxt - c.end < MIN_GAP:
            c.end = max(c.start + 0.2, nxt - MIN_GAP)
    return cues


def build_cues(words: list[Word]) -> list[Cue]:
    cues: list[Cue] = []
    cur: list[Word] = []

    def flush() -> None:
        if cur:
            cues.append(Cue(cur[0].start, cur[-1].end, _wrap(" ".join(w.text for w in cur)), cur[0].speaker))
            cur.clear()

    for w in words:
        if not w.text:
            continue
        # Whisper emits Turkish suffixes after an apostrophe as a separate token ("Serkan", "'ın")
        if cur and len(w.text) > 1 and w.text[0] in "'’":
            cur[-1] = Word(cur[-1].text + w.text, cur[-1].start, w.end, cur[-1].speaker)
            continue
        if cur:
            gap = w.start - cur[-1].end
            chars = sum(len(x.text) for x in cur) + len(cur) + len(w.text)
            last = cur[-1].text
            if (
                gap > GAP_SPLIT
                # speaker turn, but only at a sentence end or after a pause (diarizers often misplace
                # boundaries mid-sentence)
                or (w.speaker and cur[-1].speaker and w.speaker != cur[-1].speaker
                    and (last.endswith(SENT_END) or gap >= TURN_GAP))
                or chars > MAX_CHARS
                or w.end - cur[0].start > MAX_DUR
                or (last.endswith(SENT_END) and chars >= SENT_MIN)
                or (last.endswith(",") and chars >= COMMA_MIN)
            ):
                flush()
        cur.append(w)
    flush()
    return _fix_timing(_drop_repeats(cues))
