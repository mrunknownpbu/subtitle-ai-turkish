from pathlib import Path

import pytest

from subai.glossary import (GlossaryError, apply_corrections, asr_prompt, is_hallucination, detect_series_id, find_series_dir,
                            load_language, load_series)

_repo = Path(__file__).resolve().parents[1] / "glossary"
GLOSSARY = _repo if _repo.exists() else Path("/glossary")
SERIES = "tvdb-383383"


def test_detect_series_id_from_media_path():
    p = "/data/media/drama/turkish/Love Is In The Air (2020) {tvdb-383383}/Season 01/x.mkv"
    assert detect_series_id(p) == SERIES
    assert detect_series_id("/data/media/other/show.mkv") is None


def test_series_folder_contains_tvdb_id():
    d = find_series_dir(GLOSSARY, SERIES)
    assert d is not None and "tvdb-383383" in d.name


def test_series_glossary_loads_and_is_consistent():
    s = load_series(GLOSSARY, SERIES)
    assert s.meta["ids"]["tvdb"] == 383383 and s.meta["ids"]["tmdb"] == 104877
    names = {c.name for c in s.characters}
    assert {"Eda Yıldız", "Serkan Bolat", "Melek Yücel"} <= names
    assert len(s.characters) >= 30
    assert all(c.actor for c in s.characters)


def test_asr_prompt_has_leads_first_and_is_bounded():
    prompt = asr_prompt(load_series(GLOSSARY, SERIES))
    assert prompt.startswith("Eda, Serkan") and "Melo" in prompt
    assert len(prompt) <= 300


def test_language_glossary_loads():
    lang = load_language(GLOSSARY, "tr")
    assert lang["language"] == "tr"
    assert any(e["tr"] == "geçmiş olsun" for e in lang["formulae"])
    assert any(e["tr"] == "abi" for e in lang["address_terms"])


def test_corrections_whole_word_suffix_aware_and_loaded():
    s = load_series(GLOSSARY, SERIES)
    fixes = s.corrections
    assert {"heard": "Sarkan", "correct": "Serkan"}.items() <= {k: fixes[0][k] for k in ("heard", "correct")}.items()
    assert apply_corrections("Bu Sarkan Bulat'la mı?", fixes) == ("Bu Serkan Bolat'la mı?", 2)
    assert apply_corrections("Sarkan'ın Selim ve Kerim", fixes)[0] == "Serkan'ın Selim ve Kerim"
    assert apply_corrections("Sarkanlar", fixes) == ("Sarkanlar", 0)  # not a whole word


def test_hallucination_filter_matches_whole_phantom_phrases_only():
    pats = load_language(GLOSSARY, "tr")["asr_hallucinations"]
    for phantom in ("Altyazı M .K.", "Altyazı", "ALTYAZI M.K.", "İzlediğiniz için teşekkür ederim.", "Abone ol"):
        assert is_hallucination(phantom, pats), phantom
    for real in ("Bu altyazı çok güzel olmuş", "Teşekkür ederim.", "Abone değilim ben", "Evet."):
        assert not is_hallucination(real, pats), real


def test_tone_interjections_are_all_strings_including_off():
    tone = load_language(GLOSSARY, "tr")["tone"]
    assert "off" in tone["exclamation_interjections"]
    assert all(isinstance(x, str) for x in tone["exclamation_interjections"])


def test_yaml_boolean_trap_is_rejected(tmp_path):
    (tmp_path / "language").mkdir()
    (tmp_path / "language" / "tr.yaml").write_text("tone:\n  exclamation_interjections: [aa, off]\n", encoding="utf-8")
    with pytest.raises(GlossaryError, match="booleans"):
        load_language(tmp_path, "tr")


def test_missing_series_returns_none():
    assert load_series(GLOSSARY, "tvdb-1") is None


def test_bad_language_entry_rejected(tmp_path):
    (tmp_path / "language").mkdir()
    (tmp_path / "language" / "tr.yaml").write_text("formulae:\n  - {tr: x}\n", encoding="utf-8")
    with pytest.raises(GlossaryError):
        load_language(tmp_path, "tr")


@pytest.mark.parametrize("body", [
    "phrase_map:\n  - {tr: Peki}\n",
    "phrase_map:\n  - {tr: Peki, en: true}\n",  # YAML bare yes/no/true is a boolean, not a string
    "phrase_map:\n  - {tr: Peki, en: 'Okay.'}\n  - {tr: peki!, en: 'Fine.'}\n",  # same key, different output
])
def test_bad_phrase_map_rejected(tmp_path, body):
    (tmp_path / "language").mkdir()
    (tmp_path / "language" / "tr.yaml").write_text(body, encoding="utf-8")
    with pytest.raises(GlossaryError, match="phrase_map"):
        load_language(tmp_path, "tr")


def test_shipped_phrase_map_loads():
    assert load_language(GLOSSARY, "tr")["phrase_map"]
