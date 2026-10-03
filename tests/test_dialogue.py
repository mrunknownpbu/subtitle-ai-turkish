from subai.models import Cue, Word
from subai.pipeline.dialogue import assign_speakers, merge_dialogue
from subai.pipeline.segmenter import build_cues


def w(t, s, e, spk=None):
    return Word(t, s, e, spk)


SEGS = [[0.0, 2.0, "A"], [2.2, 4.0, "B"]]


def test_assign_speakers_by_overlap_and_nearest():
    ws = [w("Merhaba", 0.2, 0.8), w("selam", 2.3, 2.8), w("hmm", 4.2, 4.4), w("uzak", 30, 30.4)]
    out = assign_speakers(ws, SEGS)
    assert [x.speaker for x in out] == ["A", "B", "B", None]  # 4.2 snaps to B (0.2s away); 30s has no speaker
    assert ws[0].speaker is None  # input not mutated


def test_single_word_flip_is_smoothed():
    ws = [w("bir", 0.0, 0.3), w("iki", 0.4, 0.7), w("üç", 0.8, 1.1)]
    out = assign_speakers(ws, [[0.0, 0.35, "A"], [0.35, 0.75, "B"], [0.75, 1.2, "A"]])
    assert [x.speaker for x in out] == ["A", "A", "A"]


def test_cues_split_on_speaker_change():
    ws = [w("Merhaba", 0.0, 0.5, "A"), w("nasılsın?", 0.5, 1.0, "A"), w("İyiyim", 1.1, 1.5, "B"), w("sen?", 1.5, 1.9, "B")]
    cues = build_cues(ws)
    assert [c.text for c in cues] == ["Merhaba nasılsın?", "İyiyim sen?"]
    assert [c.speaker for c in cues] == ["A", "B"]


def test_speaker_change_mid_sentence_without_pause_is_ignored():
    # diarizer put the boundary inside "Ben tanıştırayım." -> must stay one cue, no dash
    ws = [w("Ben", 0.0, 0.3, "A"), w("tanıştırayım.", 0.35, 1.0, "B")]
    assert [c.text for c in build_cues(ws)] == ["Ben tanıştırayım."]
    # same change after a real pause is believable
    ws = [w("Ben", 0.0, 0.3, "A"), w("tanıştırayım.", 0.7, 1.4, "B")]
    assert len(build_cues(ws)) == 2


def test_dash_requires_first_speaker_to_finish_sentence():
    cues = [Cue(0.0, 1.0, "Ben", "A"), Cue(1.1, 2.0, "tanıştırayım.", "B")]
    assert len(merge_dialogue(cues)) == 2


def test_short_exchange_becomes_dash_cue():
    cues = [Cue(0.0, 1.0, "Geldin mi?", "A"), Cue(1.2, 2.0, "Geldim.", "B")]
    out = merge_dialogue(cues)
    assert len(out) == 1 and out[0].text == "- Geldin mi?\n- Geldim."
    assert (out[0].start, out[0].end) == (0.0, 2.0)


def test_no_dash_for_same_speaker_long_gap_or_long_text():
    same = [Cue(0, 1, "Bir.", "A"), Cue(1.1, 2, "İki.", "A")]
    assert len(merge_dialogue(same)) == 2
    gap = [Cue(0, 1, "Bir.", "A"), Cue(5, 6, "İki.", "B")]
    assert len(merge_dialogue(gap)) == 2
    long = [Cue(0, 2, "x" * 40, "A"), Cue(2.1, 4, "y" * 40, "B")]
    assert len(merge_dialogue(long)) == 2
