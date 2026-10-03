from subai.models import Cue
from subai.pipeline.style import apply_tone

INTERJ = ["aa", "hişt", "of", "bravo"]


def c(s, e, t):
    return Cue(s, e, t)


def test_sentence_final_interjection_gets_exclamation():
    out = apply_tone([c(0, 1, "Aa."), c(3, 4, "Evet."), c(6, 7, "Hişt. Buraya gel.")], INTERJ, continuation_dots=False)
    assert [x.text for x in out] == ["Aa!", "Evet.", "Hişt! Buraya gel."]


def test_interjection_inside_a_sentence_or_question_is_untouched():
    out = apply_tone([c(0, 1, "Aa ne güzel."), c(3, 4, "Aa?")], INTERJ, continuation_dots=False)
    assert [x.text for x in out] == ["Aa ne güzel.", "Aa?"]


def test_continuation_dots_pair_up_across_cues():
    out = apply_tone([c(0, 2, "Çıt çıkmazdı Eda çünkü"), c(2.4, 4, "bir şey olmuştu.")], [])
    assert out[0].text == "Çıt çıkmazdı Eda çünkü..."
    assert out[1].text == "...bir şey olmuştu."


def test_no_dots_after_terminal_punctuation_or_long_pause_or_dialogue_dash():
    out = apply_tone([c(0, 2, "Bitti."), c(2.2, 3, "Yeni cümle"), c(10, 11, "başka sahne"),
                      c(11.2, 12, "- Evet"), c(12.3, 13, "- Hayır")], [])
    assert [x.text for x in out] == ["Bitti.", "Yeni cümle", "başka sahne", "- Evet", "- Hayır"]


def test_input_cues_are_not_mutated():
    cues = [c(0, 1, "Aa."), c(1.2, 2, "devam")]
    apply_tone(cues, INTERJ)
    assert cues[0].text == "Aa."
