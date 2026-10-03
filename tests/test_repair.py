from subai.models import Word
from subai.pipeline.transcribe import repair_stray_words


def w(t, s, e):
    return Word(t, s, e)


def test_stray_first_word_is_moved_next_to_its_sentence():
    # real case: "Bu" stamped at 8.7 s, the rest of "Bu arada Serkanlar nerede?" at 117.3 s
    ws = [w("Bu", 8.7, 9.2), w("arada", 117.3, 117.7), w("Serkanlar", 117.7, 118.3), w("nerede?", 118.3, 118.9)]
    out = repair_stray_words(ws)
    assert out[0].text == "Bu" and 116.4 < out[0].start < out[0].end <= 117.3
    assert [x.start for x in out] == sorted(x.start for x in out)
    assert ws[0].start == 8.7  # input is not mutated


def test_stray_last_word_is_moved_back():
    ws = [w("Hadi", 10.0, 10.3), w("gel", 10.3, 10.6), w("canım", 10.6, 11.0), w("tamam", 90.0, 90.4)]
    out = repair_stray_words(ws)
    assert out[-1].text == "tamam" and out[-1].start < 12


def test_normal_pauses_and_long_halves_are_left_alone():
    ws = [w("bir", 0, 0.3), w("iki", 1.0, 1.3), w("üç", 2.0, 2.3)]
    assert [(x.start, x.end) for x in repair_stray_words(ws)] == [(0, 0.3), (1.0, 1.3), (2.0, 2.3)]
    two_halves = [w(str(i), i * 0.4, i * 0.4 + 0.3) for i in range(5)] + [w(str(i), 60 + i * 0.4, 60.3 + i * 0.4) for i in range(5)]
    assert [x.start for x in repair_stray_words(two_halves)] == [x.start for x in two_halves]
