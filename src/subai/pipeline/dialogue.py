"""Turn diarization segments into speaker-labelled words and dialogue-dash cues."""
from subai.models import Cue, Word


def assign_speakers(words: list[Word], segments: list[list], max_snap: float = 0.5) -> list[Word]:
    """Give each word the speaker whose segment overlaps it most (nearest within max_snap seconds
    if none overlaps), then smooth single-word flips between two words of the same speaker."""
    segs = sorted((float(s), float(e), str(spk)) for s, e, spk in segments)
    out: list[Word] = []
    j = 0
    for w in words:
        while j < len(segs) and segs[j][1] < w.start - 10:
            j += 1
        best, best_ov, near, near_d = None, 0.0, None, max_snap
        k = j
        while k < len(segs) and segs[k][0] <= w.end + max_snap:
            s, e, spk = segs[k]
            ov = min(w.end, e) - max(w.start, s)
            if ov > best_ov:
                best, best_ov = spk, ov
            elif ov <= 0:
                d = max(s - w.end, w.start - e)
                if d < near_d:
                    near, near_d = spk, d
            k += 1
        out.append(Word(w.text, w.start, w.end, best or near))
    for i in range(1, len(out) - 1):
        if out[i - 1].speaker and out[i - 1].speaker == out[i + 1].speaker != out[i].speaker:
            out[i].speaker = out[i - 1].speaker
    return out


def merge_dialogue(cues: list[Cue], max_gap: float = 0.8, max_chars: int = 70, max_dur: float = 6.0) -> list[Cue]:
    """Two short consecutive cues by different speakers become one two-line dash cue:
        - Merhaba.
        - Merhaba, nasılsın?
    """
    out: list[Cue] = []
    i = 0
    while i < len(cues):
        a = cues[i]
        b = cues[i + 1] if i + 1 < len(cues) else None
        if (b is not None and a.speaker and b.speaker and a.speaker != b.speaker
                and a.text.rstrip().endswith((".", "?", "!", "…"))  # first speaker finished a sentence
                and "\n" not in a.text and "\n" not in b.text
                and b.start - a.end <= max_gap
                and len(a.text) + len(b.text) <= max_chars
                and b.end - a.start <= max_dur):
            out.append(Cue(a.start, b.end, f"- {a.text}\n- {b.text}"))
            i += 2
        else:
            out.append(Cue(a.start, a.end, a.text, a.speaker))
            i += 1
    return out
