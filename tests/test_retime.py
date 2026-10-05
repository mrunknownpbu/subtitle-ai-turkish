import random
from dataclasses import dataclass

import pytest

from subai import retime
from subai.models import Word
from subai.retime import RetimeRefused


@dataclass
class Cue:
    start: float
    end: float
    text: str


def make_programme(n=400, seed=3, vocab_size=600):
    """Ground-truth cues and the audio's words for them (words spread inside each cue)."""
    rng = random.Random(seed)
    vocab = [f"kelime{i}" for i in range(vocab_size)]
    cues, words, t = [], [], 5.0
    for _ in range(n):
        text = " ".join(rng.choice(vocab) for _ in range(rng.randint(4, 9)))
        duration = rng.uniform(1.5, 3.5)
        cues.append(Cue(t, t + duration, text))
        pieces = text.split()
        for k, w in enumerate(pieces):
            a = t + duration * k / len(pieces)
            words.append(Word(w, a, a + duration / len(pieces)))
        t += duration + rng.uniform(0.3, 3.0)
    return cues, words


def distort(cues, fn):
    return [Cue(fn(c.start), fn(c.end), c.text) for c in cues]


def mean_error(truth, times):
    return sum(abs(a.start - s) + abs(a.end - e) for a, (s, e) in zip(truth, times)) / (2 * len(truth))


@pytest.fixture
def prog():
    return make_programme()


def check(prog, fn, method, limit=0.2):
    truth, words = prog
    times, report = retime.retime(distort(truth, fn), words)
    assert report.method == method
    assert mean_error(truth, times) < limit
    return report


def test_constant_shift_late(prog):
    report = check(prog, lambda t: t + 5.1, "constant offset")
    assert report.pieces[0].offset0 == pytest.approx(-5.1, abs=0.1)


def test_constant_shift_early(prog):
    check(prog, lambda t: t - 3.4, "constant offset")


def test_frame_rate_drift(prog):  # 25 fps subtitle against 23.976 fps audio
    assert len(check(prog, lambda t: t * 23.976 / 25, "linear drift").pieces) == 1


def test_step_from_a_cut(prog):
    mid = prog[0][len(prog[0]) // 2].start
    assert len(check(prog, lambda t: t + (4.0 if t > mid else 0.0), "stepped").pieces) == 2


def test_two_steps(prog):
    a, b = prog[0][130].start, prog[0][270].start
    check(prog, lambda t: t + (0 if t < a else 2.5 if t < b else -1.5), "stepped", limit=0.4)


def test_already_aligned_changes_nothing(prog):
    truth, words = prog
    times, report = retime.retime(truth, words)
    assert report.method == "aligned" and not report.changed
    assert times == [(c.start, c.end) for c in truth]


def test_noisy_cue_times_still_fit(prog):
    truth, words = prog
    rng = random.Random(9)
    noisy = [Cue(c.start + 5 + rng.uniform(-.4, .4), c.end + 5 + rng.uniform(-.4, .4), c.text) for c in truth]
    times, report = retime.retime(noisy, words)
    assert report.method == "constant offset"
    assert mean_error(truth, times) < 0.5


def test_survives_wrong_matches(prog):
    truth, words = prog
    rng = random.Random(4)
    cues = distort(truth, lambda t: t + 5.1)
    for c in rng.sample(cues, 30):  # 7% of cues carry other text
        c.text = " ".join(f"zzz{rng.randint(0, 99)}" for _ in range(6))
    times, report = retime.retime(cues, words)
    assert report.method == "constant offset"
    good = [(a, t) for a, t, c in zip(truth, times, cues) if not c.text.startswith("zzz")]
    assert mean_error([a for a, _ in good], [t for _, t in good]) < 0.2


def test_refuses_a_different_programme(prog):
    truth, _ = prog
    _, other_words = make_programme(seed=99)
    other = [Word(f"baska{i % 700}", w.start, w.end) for i, w in enumerate(other_words)]
    with pytest.raises(RetimeRefused):
        retime.retime(distort(truth, lambda t: t + 5), other)


def test_refuses_nonsense_disagreement(prog):
    truth, words = prog
    cues = distort(truth, lambda t: t + 3)
    for i, c in enumerate(cues):  # alternate stretches disagree wildly: no single story
        if i % 2:
            c.start += 40 * ((i // 2) % 3)
            c.end += 40 * ((i // 2) % 3)
    with pytest.raises(RetimeRefused):
        retime.retime(cues, words)


def test_cues_never_start_before_zero_or_overlap_after_a_step(prog):
    truth, words = prog
    mid = truth[200].start
    times, _ = retime.retime(distort(truth, lambda t: t + (-4.0 if t > mid else 3.0)), words)
    assert all(s >= 0 for s, _ in times)
    assert all(e0 <= s1 for (_, e0), (s1, _) in zip(times, times[1:]))


def test_tokenisation_ignores_tags_case_and_punctuation():
    assert retime.tokens("<i>Nasılsın</i>, IYIYIM! (gülüyor)", "tr") == ["nasılsın", "iyiyim"]


def test_retime_file_uses_cached_words_and_keeps_text(tmp_path):
    import json

    import pysrt

    from subai.pipeline import runner

    truth, words = make_programme()
    video = tmp_path / "ep.mkv"
    video.write_bytes(b"x")
    (tmp_path / "out" / ".subai").mkdir(parents=True)
    (tmp_path / "out" / ".subai" / "ep.words.json").write_text(json.dumps(
        {"key": {}, "duration": 1, "words": [[w.text, w.start, w.end] for w in words]}), encoding="utf-8")
    texts = [c.text.replace(" ", "\n", 1) for c in truth]  # a line break must survive
    src = tmp_path / "in.srt"
    pysrt.SubRipFile(items=[pysrt.SubRipItem(i + 1, start=round((c.start + 5.1) * 1000), end=round((c.end + 5.1) * 1000),
                                             text=t) for i, (c, t) in enumerate(zip(truth, texts))]).save(str(src), encoding="utf-8")

    out = tmp_path / "res" / "out.srt"
    report, source = runner.retime_file(video, src, out, lambda: pytest.fail("must not transcribe"), tmp_path)

    got = pysrt.open(str(out), encoding="utf-8")
    assert source == "cache" and report.method == "constant offset"
    assert [s.text for s in got] == texts
    assert sum(abs(s.start.ordinal / 1000 - c.start) for s, c in zip(got, truth)) / len(truth) < 0.2
    assert not out.with_name("out.srt.part").exists()
