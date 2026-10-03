from subai.models import Word
from subai.pipeline.segmenter import MAX_LINE, build_cues


def w(t, s, e):
    return Word(t, s, e)


def test_splits_on_long_pause():
    cues = build_cues([w("Merhaba", 0, 0.5), w("dünya", 0.5, 1.0), w("Nasılsın", 3.0, 3.6)])
    assert len(cues) == 2
    assert cues[0].text == "Merhaba dünya"


def test_line_length_limit():
    words = [w("kelime" + str(i), i * 0.3, i * 0.3 + 0.25) for i in range(14)]
    for c in build_cues(words):
        assert all(len(line) <= MAX_LINE + 6 for line in c.text.split("\n"))
        assert len(c.text.split("\n")) <= 2


def test_min_duration_and_no_overlap():
    cues = build_cues([w("Evet.", 0, 0.2), w("Hayır", 1.0, 1.4)])
    assert cues[0].end - cues[0].start >= 0.3
    assert cues[0].end <= cues[1].start


def test_fast_cue_is_extended_into_following_silence_for_readability():
    # 34 chars spoken in 0.8 s, then 3 s of silence: must be held ~2 s (17 cps), not cut at the last word
    cues = build_cues([w("Ağaçları", 0, 0.4), w("getireceğiz", 0.4, 0.8), w("daha", 0.8, 1.0), w("Sonra", 4.5, 5.0)])
    c = cues[0]
    assert c.end - c.start >= len(c.text) / 17 - 0.01
    assert c.end <= cues[1].start - 0.08 + 1e-9


def test_extension_never_overlaps_next_cue():
    cues = build_cues([w("Çok", 0, 0.3), w("hızlı", 0.3, 0.6), w("konuşuyorsun", 0.7, 1.4), w("hadi", 1.5, 1.9)])
    for a, b in zip(cues, cues[1:]):
        assert a.end <= b.start


def test_joins_apostrophe_suffix():
    cues = build_cues([w("Ben", 0, 0.3), w("Serkan", 0.3, 0.8), w("'ın", 0.8, 1.0), w("sevgilisiyim.", 1.0, 1.8)])
    assert cues[0].text == "Ben Serkan'ın sevgilisiyim."


def test_drops_hallucination_loop():
    words = []
    for i in range(6):
        words.append(w("Altyazı", i * 2.0, i * 2.0 + 0.5))
    assert len(build_cues(words)) <= 2
