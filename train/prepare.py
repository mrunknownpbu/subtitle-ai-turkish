#!/usr/bin/env python3
"""Build a Whisper fine-tuning set: OUR transcript, corrected by the human reference where it clearly
fixes a substantive word.

Why not train on the reference text directly: it is condensed and written-standard (drops fillers,
"vallahi" for spoken "valla"), so it would teach the model to lose the natural spoken tone.
Merge rules per word, after aligning reference and hypothesis (spelling variants count as equal):
  ok                      -> keep our word (with our punctuation and casing)
  sub (different word)    -> take the reference word, unless either side is a discourse filler
  ins (we have extra)     -> keep ours (assume real speech, e.g. "ya", "yani")
  del (reference extra)   -> add it only if it is a short backchannel Whisper often drops ("evet", "tamam" ...)
Windows are skipped if alignment is poor (content WER > 0.4), or they contain song lyrics / music notes.

Run inside the training image with the repo mounted at /repo and /data mounted read-only:
  python /repo/train/prepare.py --episodes 1 2 3 --out /repo/workspace/train
"""
import argparse
import json
import re
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))
import wer  # noqa: E402

MEDIA = Path("/data/media/drama/turkish/Love Is In The Air (2020) {tvdb-383383}/Season 01")
REF_DIR = MEDIA / "tr_sub"
BACKCHANNEL = {"evet", "hayır", "tamam", "yok", "olur", "peki", "bak", "hadi", "gel", "tabii", "anladım", "hı"}
LYRICS = re.compile(r"♪|çalıyor|şarkı|müzik", re.I)
PUNCT_END = re.compile(r"[.,!?…]+$")


def canon(t: str) -> str:
    return wer.VARIANTS.get(t, t)


def surface_tokens(words: list[list]) -> list[str]:
    """Whisper emits Turkish suffixes after an apostrophe as separate tokens; re-join them."""
    out: list[str] = []
    for text, *_ in words:
        if out and text[:1] in "'’" and len(text) > 1:
            out[-1] += text
        else:
            out.append(text)
    return out


def _items(surfaces: list[str]) -> list[tuple[str, str]]:
    """(canonical word, surface) for each whitespace token that carries a word; stray punctuation
    tokens stick to the previous word."""
    items: list[tuple[str, str]] = []
    for s in surfaces:
        n = wer.tokens(s)
        if n:
            items.append((canon("".join(n) if len(n) > 1 else n[0]), s))
        elif items:
            items[-1] = (items[-1][0], items[-1][1] + s)
    return items


def _core(surface: str) -> str:
    return PUNCT_END.sub("", surface).strip("\"'“”()-–")


def merge(ref_text: str, hyp_surface: list[str], keep_ref_punct: bool = True) -> tuple[str, float]:
    """Words from our transcript (ref word where it clearly fixes one); with keep_ref_punct the
    reference's punctuation and casing are used, since it is professionally punctuated."""
    ritems, hitems = _items(ref_text.split()), _items(hyp_surface)
    ops = wer.align([k for k, _ in ritems], [k for k, _ in hitems])
    cwer = sum(o[0] != "ok" for o in ops) / max(len(ritems), 1)
    out, hi, ri = [], 0, 0
    for op, r, h in ops:
        if op == "ok":
            hs, rs = hitems[hi][1], ritems[ri][1]
            hi += 1; ri += 1
            if keep_ref_punct:
                word = _core(hs)
                word = (word[:1].upper() if _core(rs)[:1].isupper() else word[:1].lower()) + word[1:]
                m = PUNCT_END.search(rs)
                out.append(word + (m.group(0) if m else ""))
            else:
                out.append(hs)
        elif op == "ins":
            out.append(hitems[hi][1]); hi += 1
        elif op == "sub":
            hs, rs = hitems[hi][1], ritems[ri][1]
            hi += 1; ri += 1
            out.append(hs if (r in wer.FILLERS or h in wer.FILLERS) else rs.strip("\"'“”()-–"))
        elif op == "del":
            rs = ritems[ri][1]; ri += 1
            if r in BACKCHANNEL:
                out.append(rs.strip("\"'“”()-–"))
    return " ".join(out), cwer


def join_cues(texts: list[str]) -> str:
    """Join subtitle cues into running text: drop the '...' that subtitlers put at both sides of a cue
    break and the dialogue dashes; they are subtitle conventions, not speech."""
    out = ""
    for t in texts:
        t = wer.ANNOT.sub(" ", t).strip()
        t = re.sub(r"(^|\s)[-–]\s*", r"\1", t)
        if out.endswith(("...", "…")) and t.startswith(("...", "…")):
            out = re.sub(r"(\.\.\.|…)$", "", out).rstrip()
            t = re.sub(r"^(\.\.\.|…)\s*", "", t)
        out = (out + " " + t).strip()
    return out


def write_wav(path: Path, samples: bytes) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
        w.writeframes(samples)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--words-dir", type=Path, default=Path("/repo/workspace/output/tvdb-383383/S01/.subai"))
    ap.add_argument("--max-window", type=float, default=28.0)
    ap.add_argument("--max-cwer", type=float, default=0.4)
    ap.add_argument("--hyp-punct", action="store_true", help="keep OUR punctuation instead of the reference's")
    a = ap.parse_args()
    (a.out / "audio").mkdir(parents=True, exist_ok=True)
    refs = {int(re.search(r"(\d)\.\s?B", f.name)[1]): f for f in REF_DIR.glob("*.srt")}
    rows, skipped = [], {"lyrics": 0, "poor_alignment": 0, "short": 0}
    for ep in a.episodes:
        name = f"Love Is In The Air (2020) S01E{ep:02d}"
        words = json.load(open(a.words_dir / f"{name}.words.json", encoding="utf-8"))["words"]
        cues = [(s, e, t) for s, e, t in wer.parse_srt(refs[ep])]
        with tempfile.TemporaryDirectory() as tmp:
            wav_path = Path(tmp) / "a.wav"
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(MEDIA / f"{name}.mkv"), "-vn", "-ac", "1",
                            "-ar", "16000", "-c:a", "pcm_s16le", str(wav_path)], check=True)
            with wave.open(str(wav_path), "rb") as w:
                pcm = w.readframes(w.getnframes())
        windows, cur = [], []
        for c in cues:
            if cur and (c[1] - cur[0][0] > a.max_window or c[0] - cur[-1][1] > 6):
                windows.append(cur); cur = []
            cur.append(c)
        if cur:
            windows.append(cur)
        for i, win in enumerate(windows):
            ws, we = max(win[0][0] - 0.3, 0), win[-1][1] + 0.3
            raw = " ".join(t for _, _, t in win)
            if LYRICS.search(raw) or '"' in raw:
                skipped["lyrics"] += 1; continue
            ref_text = join_cues([t for _, _, t in win])
            hw = [w for w in words if ws <= (w[1] + w[2]) / 2 <= we]
            if len(hw) < 3 or len(wer.tokens(ref_text)) < 3:
                skipped["short"] += 1; continue
            label, cwer = merge(ref_text, surface_tokens(hw), keep_ref_punct=not a.hyp_punct)
            if cwer > a.max_cwer:
                skipped["poor_alignment"] += 1; continue
            p = a.out / "audio" / f"E{ep:02d}_{i:04d}.wav"
            write_wav(p, pcm[int(ws * 16000) * 2: int(we * 16000) * 2])
            rows.append({"audio": str(p), "text": label, "ref": ref_text.strip(), "episode": ep,
                         "start": round(ws, 2), "end": round(we, 2), "cwer": round(cwer, 3)})
        print(f"E{ep:02d}: {len(windows)} windows -> kept so far {len(rows)}", flush=True)
    with open(a.out / "train.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    hours = sum(r["end"] - r["start"] for r in rows) / 3600
    print(f"\nkept {len(rows)} windows = {hours:.2f} h; skipped {skipped}")
    for r in rows[:: max(len(rows) // 6, 1)][:6]:
        print(f"\n[E{r['episode']:02d} {r['start']:.0f}s cwer={r['cwer']}]\n  REF  : {r['ref'][:150]}\n  LABEL: {r['text'][:150]}")


if __name__ == "__main__":
    main()
