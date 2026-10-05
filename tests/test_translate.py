import pysrt
import pytest

from subai import protect as P
from subai import translate as T
from subai.glossary import Character, SeriesGlossary


def write_srt(path, texts):
    subs = pysrt.SubRipFile(items=[
        pysrt.SubRipItem(i + 1, start=i * 2000, end=i * 2000 + 1500, text=t) for i, t in enumerate(texts)])
    subs.save(str(path), encoding="utf-8")


def fake_mt():
    """Stand-in for the model: wraps every input, keeps placeholders intact, records each call's payload."""
    calls = []

    def mt(xs):
        calls.append(list(xs))
        return [f"EN({x})" for x in xs]

    mt.calls = calls
    return mt


def series():
    return SeriesGlossary("tvdb-1", {}, {}, [Character("Serkan Bolat", ["Serkan", "Serkan Bey"], "A", "male", "lead")],
                          [{"tr": "burs", "en": "scholarship", "policy": "translate"},
                           {"tr": "Art Life", "en": "Art Life", "policy": "keep"}])


def test_fix_honorifics_swaps_madam_and_title_order():
    assert T.fix_honorifics("Ayfer Hanım", "Madam Ayfer") == "Ayfer Hanım"
    assert T.fix_honorifics("Evren Bey'in dosyası", "Mr. Evren's file") == "Evren Bey's file"
    assert T.fix_honorifics("Evren Bey'in dosyası", "the file of Bey Evren.") == "the file of Evren Bey."
    assert T.fix_honorifics("Gel buraya.", "Mr. Smith, come here.") == "Mr. Smith, come here."  # no Bey/Hanım in source


def test_wrap_only_long_single_line_without_dash():
    assert T.wrap("short") == "short"
    assert "\n" in T.wrap("word " * 20)
    assert "\n" not in T.wrap("- " + "word " * 20)
    assert T.wrap("a\nb") == "a\nb"


def test_collapse_repeats():
    assert T.collapse_repeats("Aşkım, sen gel, gel, gel, gel, gel, gel...") == "Aşkım, sen gel gel..."
    assert T.collapse_repeats("Gel gel.") == "Gel gel."


def test_protect_restore_turkish_names():
    g = P.build_glossary([P.Entity("Istanbul", ["İstanbul"]), P.Entity("Serkan", ["Serkan"])])
    assert list(g) == ["İstanbul", "Serkan"]  # keys keep the original spelling (casefold would break İ)
    p = P.protect("İstanbul'da SERKAN'ın evi", g)
    assert "İstanbul" not in p and "SERKAN" not in p and "'da" in p
    assert P.restore(p, g) == "Istanbul'da Serkan'ın evi"


def test_longest_form_wins_and_short_forms_skipped():
    g = P.build_glossary([P.Entity("Serkan Bolat", ["Serkan Bolat"]), P.Entity("Serkan", ["Serkan", "Se"])])
    assert "Se" not in g
    assert P.restore(P.protect("Serkan Bolat geldi", g), g) == "Serkan Bolat geldi"
    assert len(g["Serkan Bolat"][0]) == 3 and g["Serkan Bolat"][0] != g["Serkan"][0]


def test_repair_corrupted_placeholders_only_when_unambiguous():
    assert P.repair_corrupted_placeholders("hi Xax", "gel Xac") == "hi Xac"
    assert P.repair_corrupted_placeholders("hi Xax", "Xac ve Xad") == "hi Xax"  # two candidates: leave it


def test_bare_entity_skips_the_model():
    g = P.build_glossary([P.Entity("Cenk", ["Cenk"])])
    assert P.bare_entity_translation(P.protect("Cenk.", g), g) == "Cenk."
    assert P.bare_entity_translation(P.protect("Cenk geldi.", g), g) is None


def test_recover_dropped_entities_prepends_the_name():
    g = P.build_glossary([P.Entity("Cenk", ["Cenk"])])
    src = P.protect("Cenk, gel!", g)
    assert P.recover_dropped_entities(src, "Come here!", g) == "Cenk! Come here!"
    assert P.recover_dropped_entities(src, "Cenk, come!", g) == "Cenk, come!"


def test_split_dash_lines_and_sentences():
    assert P.split_dash_lines("- a\n- b") == ["a", "b"]
    assert P.split_dash_lines("- a\nb") is None and P.split_dash_lines("- a") is None
    assert P.split_sentences("Bir. İki!") == ["Bir.", "İki!"]
    assert P.split_sentences("Tek cümle.") is None


def test_run_on_and_chunks():
    text = "bir iki üç dört beş altı yedi sekiz dokuz"
    assert P.is_run_on(text) and not P.is_run_on(text + ".") and not P.is_run_on("bir iki")
    assert P.chunk_words(text) == ["bir iki üç dört beş altı.", "yedi sekiz dokuz."]


def test_phrase_key_handles_turkish_dotted_i():
    assert P.phrase_key("İyi!") == P.phrase_key("iyi") == "iyi"
    assert P.phrase_key("Işık?") == "ışık"


def test_qc_flag():
    assert P.qc_flag("Gel buraya.", "") == "empty"
    assert P.qc_flag("Gel buraya.", "Xaa left") == "placeholder leaked"
    assert P.qc_flag("Gel buraya.", "(Come here.") == "unbalanced bracket"
    assert P.qc_flag("x" * 40, "ok") == "too short"
    assert P.qc_flag("ab", "x" * 20) == "too long"
    assert P.qc_flag("Gel buraya.", "Come here.") is None


def test_translate_cues_phrase_map_names_and_terms():
    mt = fake_mt()
    g = T.build_protection(series())
    out = T.translate_cues(["Peki!", "Serkan Bey burs verdi.", "Serkan."], mt, g, {"peki": "Okay."})
    assert out[0] == "Okay!"  # phrase map answers, punctuation follows the source
    assert out[1] == "EN(Serkan Bey scholarship verdi.)"  # name kept, fixed term translated
    assert out[2] == "Serkan."  # bare name never reaches the model
    assert len(mt.calls) == 1 and len(mt.calls[0]) == 1
    assert "Serkan" not in mt.calls[0][0] and "burs" not in mt.calls[0][0]  # the model only saw placeholders


def test_translate_cues_splits_dash_turns_and_sentences():
    mt = fake_mt()
    out = T.translate_cues(["- Gel.\n- Gelmem.", "Bir. İki."], mt, {}, {})
    assert out == ["- EN(Gel.)\n- EN(Gelmem.)", "EN(Bir.) EN(İki.)"]
    assert mt.calls == [["Gel.", "Gelmem.", "Bir.", "İki."]]  # one batched call


def test_run_on_is_retried_in_chunks_and_better_result_kept():
    text = "bir iki üç dört beş altı yedi sekiz dokuz"
    calls = []

    def mt(xs):
        calls.append(list(xs))
        return ["x" if t == text else f"w({t})" for t in xs]  # the whole run-on comes back nearly empty

    out = T.translate_cues([text], mt, {}, {})
    assert calls[1] == ["bir iki üç dört beş altı.", "yedi sekiz dokuz."]
    assert out == ["w(bir iki üç dört beş altı.) w(yedi sekiz dokuz.)"]


def test_build_phrase_map():
    assert T.build_phrase_map({"phrase_map": [{"tr": "Peki", "en": "Okay."}]}) == {"peki": "Okay."}
    assert T.build_phrase_map(None) == {}


def test_translate_srt_end_to_end(tmp_path):
    src, dst = tmp_path / "a.tr.srt", tmp_path / "a.en.srt"
    write_srt(src, ["Ayfer Hanım.", "- Gel.\n- Gelmem.", "Tamam."])
    mt = lambda xs: [{"Ayfer Hanım.": "Madam Ayfer."}.get(x, x.upper()) for x in xs]
    assert T.translate_srt(src, dst, None, translate_fn=mt, phrase_map={"tamam": "Okay."}) == 3
    out = pysrt.open(str(dst), encoding="utf-8")
    assert [s.text for s in out] == ["Ayfer Hanım.", "- GEL.\n- GELMEM.", "Okay."]
    assert [(s.start.ordinal, s.end.ordinal) for s in out] == [(i * 2000, i * 2000 + 1500) for i in range(3)]
    assert not (tmp_path / "a.en.srt.part").exists()


def test_translator_needs_a_gpu(monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="GPU"):
        T.Translator()


def test_fix_gender_swaps_the_guessed_word_only_when_the_stem_is_in_the_source():
    sg = SeriesGlossary("t", {}, {}, [], [{"tr": "torun", "en": "granddaughter", "wrong": "grandson", "policy": "gender_fix"}])
    assert T.fix_gender("O benim torunum.", "He's my grandson.", sg) == "He's my granddaughter."
    assert T.fix_gender("Torunumu aldılar.", "Grandsons first.", sg) == "Granddaughters first."
    assert T.fix_gender("Oğlum geldi.", "My grandson came.", sg) == "My grandson came."  # no torun in the source
    assert T.fix_gender("torunum", "grandson", None) == "grandson"
    assert T.build_protection(sg) == {}  # a gender_fix term is not a protected name
