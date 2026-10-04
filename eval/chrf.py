#!/usr/bin/env python3
"""Score an English SRT against a human English SRT with corpus chrF (6-grams, beta 2).

Each of our cues is paired with the reference cues that lie mostly inside it; song lyrics,
sound descriptions and unmatched cues are skipped. The reference (`*.en.hi.srt`) is for
scoring only, never pipeline input. Method follows subtitle-ai/scripts/eval_translation.py.

    python3 eval/chrf.py --ref REF.en.hi.srt --hyp A.en.srt [--hyp B.en.srt ...] [--verbose N]
"""
import argparse
import random
import re
from collections import Counter

ORDER, BETA = 6, 2.0
_TS = re.compile(r"(\d+):(\d\d):(\d\d)[,.](\d{3})\s*-->\s*(\d+):(\d\d):(\d\d)[,.](\d{3})")
_SDH = re.compile(r"\([^)]*\)|\[[^\]]*\]|♪|♫")


def read_srt(path):
    cues = []
    for block in re.split(r"\n\s*\n", open(path, encoding="utf-8-sig").read().strip()):
        lines = block.strip().splitlines()
        for i, line in enumerate(lines):
            m = _TS.search(line)
            if m:
                g = list(map(int, m.groups()))
                cues.append((g[0] * 3600 + g[1] * 60 + g[2] + g[3] / 1000,
                             g[4] * 3600 + g[5] * 60 + g[6] + g[7] / 1000, " ".join(lines[i + 1:])))
                break
    return cues


def clean(text):
    text = _SDH.sub(" ", re.sub(r"<[^>]+>", "", text))
    return re.sub(r"\s+", " ", re.sub(r"(^|\s)-\s*", " ", text)).strip()


def pair(hyp, ref):
    """(hyp text, ref text) per hyp cue whose reference is identifiable."""
    out = []
    for s, e, text in hyp:
        refs, covered = [], 0.0
        for rs, re_, rt in ref:
            overlap = min(e, re_) - max(s, rs)
            if overlap > 0 and overlap >= 0.5 * max(re_ - rs, 1e-6):
                refs.append(rt)
                covered += overlap
        raw = " ".join(refs)
        if not refs or covered < 0.5 * max(e - s, 1e-6) or raw.lstrip("-( ").startswith(('"', "“", "♪")):
            continue
        if clean(raw) and clean(text):
            out.append((clean(text), clean(raw)))
    return out


def grams(t, n):
    t = t.replace(" ", "")
    return Counter(t[i:i + n] for i in range(len(t) - n + 1))


def stats(h, r):
    res = []
    for n in range(1, ORDER + 1):
        a, b = grams(h, n), grams(r, n)
        res.append((sum((a & b).values()), sum(a.values()), sum(b.values())))
    return res


def chrf(st):
    ps, rs = [], []
    for n in range(ORDER):
        m, h, r = (sum(s[n][k] for s in st) for k in range(3))
        if h and r:
            ps.append(m / h)
            rs.append(m / r)
    p, r = sum(ps) / max(len(ps), 1), sum(rs) / max(len(rs), 1)
    return 100 * (1 + BETA ** 2) * p * r / (BETA ** 2 * p + r) if p + r else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--hyp", action="append", required=True)
    ap.add_argument("--verbose", type=int, default=0, help="show N worst pairs per hypothesis")
    a = ap.parse_args()
    ref = read_srt(a.ref)
    base = None
    for path in a.hyp:
        pairs = pair(read_srt(path), ref)
        st = [stats(h, r) for h, r in pairs]
        score = chrf(st)
        line = f"{score:6.2f} chrF  {len(pairs):5d} pairs  {path}"
        if base is None:
            base = (st, score)
        elif len(st) == len(base[0]):  # same cues -> same pairs: paired bootstrap vs the first
            rng, diffs = random.Random(0), []
            for _ in range(300):
                idx = [rng.randrange(len(st)) for _ in st]
                diffs.append(chrf([st[i] for i in idx]) - chrf([base[0][i] for i in idx]))
            diffs.sort()
            line += f"  vs first {score - base[1]:+.2f} [{diffs[7]:+.2f}, {diffs[292]:+.2f}]"
        print(line)
        for s, (h, r) in sorted(zip(st, pairs), key=lambda x: chrf([x[0]]))[:a.verbose]:
            print(f"   {chrf([s]):5.1f}  HYP {h}\n         REF {r}")


if __name__ == "__main__":
    main()
