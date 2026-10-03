#!/usr/bin/env python3
"""Score a hypothesis SRT against a reference SRT (word error rate, by time window).

Usage: python3 eval/wer.py --ref REF.srt --hyp HYP.srt [--window 60] [--top 25] [--keep-annotations]

Reference subtitles for the hearing impaired contain speaker labels "(Serkan)" and sound notes
"(müzik)"; these are stripped by default because they are not spoken words. Words are lowercased
with Turkish rules (İ->i, I->ı), punctuation removed, apostrophes joined (Serkan'ın -> serkanın)
and circumflexes dropped (hâlâ -> hala).

Two scores are reported:
  strict  - every word counts.
  content - spoken/written spelling variants are unified (vallahi=valla, bayağı=baya, tabii=tabi ...)
            and discourse fillers (ya, yani, ay, ee, aa ...) are ignored. Reference subtitles are
            condensed and omit fillers, so this is the fair number to tune against.

Alignment: tokens are binned by cue midpoint into time windows and edit-distance is computed per
window, so a timing drift never lets one early error cascade across the whole episode.
"""
import argparse
import collections
import re
from pathlib import Path

TS = re.compile(r"(\d+):(\d+):(\d+)[,.](\d+) --> (\d+):(\d+):(\d+)[,.](\d+)")
ANNOT = re.compile(r"\([^)]*\)|\[[^\]]*\]|♪[^♪]*♪|♪")
CIRC = str.maketrans("âîû", "aiu")
VARIANTS = {"vallahi": "valla", "bayağı": "baya", "tabii": "tabi", "abiciğim": "abicim", "ayy": "ay",
            "ayyy": "ay", "eee": "ee", "ııı": "ıı", "yoo": "yo", "okay": "okey", "hımm": "hım"}
FILLERS = {"ya", "yani", "ay", "ee", "ıı", "ha", "hı", "hım", "aa", "e", "ah", "of", "eh", "oo", "ıh", "mm", "hmm"}


def read_text(path: Path) -> str:
    raw = path.read_bytes()
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        return raw.decode("cp1254")


def parse_srt(path: Path) -> list[tuple[float, float, str]]:
    cues = []
    for blk in re.split(r"\r?\n\r?\n+", read_text(path).strip()):
        lines = [l for l in blk.splitlines() if l.strip()]
        for i, l in enumerate(lines):
            m = TS.search(l)
            if m:
                g = list(map(int, m.groups()))
                s = g[0] * 3600 + g[1] * 60 + g[2] + g[3] / 1000
                e = g[4] * 3600 + g[5] * 60 + g[6] + g[7] / 1000
                text = re.sub(r"<[^>]+>", "", " ".join(lines[i + 1:]))
                cues.append((s, e, text))
                break
    return cues


def tokens(text: str, keep_annotations: bool = False, content: bool = False) -> list[str]:
    if not keep_annotations:
        text = ANNOT.sub(" ", text)
    text = text.replace("İ", "i").replace("I", "ı").lower().translate(CIRC)
    text = re.sub(r"['’`]", "", text)
    text = re.sub(r"[^\w\s]|_", " ", text)
    toks = text.split()
    if content:
        toks = [VARIANTS.get(t, t) for t in toks]
        toks = [t for t in toks if t not in FILLERS]
    return toks


def align(ref: list[str], hyp: list[str]):
    """Levenshtein with backtrace. Returns list of (op, ref_word, hyp_word)."""
    n, m = len(ref), len(hyp)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]))
    ops, i, j = [], n, m
    while i or j:
        if i and j and d[i][j] == d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]):
            ops.append(("ok" if ref[i - 1] == hyp[j - 1] else "sub", ref[i - 1], hyp[j - 1]))
            i, j = i - 1, j - 1
        elif i and d[i][j] == d[i - 1][j] + 1:
            ops.append(("del", ref[i - 1], ""))
            i -= 1
        else:
            ops.append(("ins", "", hyp[j - 1]))
            j -= 1
    return ops[::-1]


def bin_tokens(cues, window: float, keep: bool = False, content: bool = False) -> dict[int, list[str]]:
    bins: dict[int, list[str]] = collections.defaultdict(list)
    for s, e, t in cues:
        bins[int(((s + e) / 2) // window)].extend(tokens(t, keep, content))
    return bins


def score(ref_cues, hyp_cues, window: float = 60.0, keep: bool = False, content: bool = False) -> dict:
    """Importable scorer. ref/hyp are lists of (start, end, text). Returns counts and error tables."""
    R = bin_tokens(ref_cues, window, keep, content)
    H = bin_tokens(hyp_cues, window, keep, content)
    out = {"N": 0, "S": 0, "D": 0, "I": 0, "hyp_words": sum(len(v) for v in H.values()),
           "subs": collections.Counter(), "dels": collections.Counter(), "inss": collections.Counter(), "windows": []}
    for b in sorted(set(R) | set(H)):
        ops = align(R.get(b, []), H.get(b, []))
        s = d = i = 0
        for op, r, h in ops:
            if op == "sub":
                s += 1
                out["subs"][(r, h)] += 1
            elif op == "del":
                d += 1
                out["dels"][r] += 1
            elif op == "ins":
                i += 1
                out["inss"][h] += 1
        n = len(R.get(b, []))
        out["N"] += n
        out["S"] += s
        out["D"] += d
        out["I"] += i
        out["windows"].append(((s + d + i) / max(n, 1), b, n, s, d, i))
    out["wer"] = (out["S"] + out["D"] + out["I"]) / max(out["N"], 1)
    return out


def report(r: dict, label: str) -> str:
    N = max(r["N"], 1)
    return (f"{label:8s} WER={r['wer']:.1%}  sub={r['S'] / N:.1%}  del={r['D'] / N:.1%}  ins={r['I'] / N:.1%}"
            f"  (ref words={r['N']}, hyp words={r['hyp_words']})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", type=Path, required=True)
    ap.add_argument("--hyp", type=Path, required=True)
    ap.add_argument("--window", type=float, default=60.0)
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--keep-annotations", action="store_true")
    a = ap.parse_args()
    ref, hyp = parse_srt(a.ref), parse_srt(a.hyp)
    strict = score(ref, hyp, a.window, a.keep_annotations, False)
    content = score(ref, hyp, a.window, a.keep_annotations, True)
    print(report(strict, "strict"))
    print(report(content, "content"))
    print(f"\nworst {a.window:.0f}s windows by content WER (wer, start, ref_words, S/D/I):")
    for w, b, n, s, d, i in sorted(content["windows"], reverse=True)[:8]:
        t = int(b * a.window)
        print(f"  {w:5.0%}  {t // 3600}:{t % 3600 // 60:02d}:{t % 60:02d}  n={n}  {s}/{d}/{i}")
    print("\ntop substitutions (ref -> hyp), content mode:")
    print("  " + ", ".join(f"{r}->{h} x{c}" for (r, h), c in content["subs"].most_common(a.top)))
    print("top deletions (in reference, missing from hyp):")
    print("  " + ", ".join(f"{w} x{c}" for w, c in content["dels"].most_common(a.top)))
    print("top insertions (in hyp, not in reference):")
    print("  " + ", ".join(f"{w} x{c}" for w, c in content["inss"].most_common(a.top)))


if __name__ == "__main__":
    main()
