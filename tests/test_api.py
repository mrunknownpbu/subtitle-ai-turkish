import pysrt
import pytest
from fastapi import HTTPException

from subai import api


def write_srt(path, texts):
    pysrt.SubRipFile(items=[pysrt.SubRipItem(i + 1, start=i * 2000, end=i * 2000 + 1500, text=t)
                            for i, t in enumerate(texts)]).save(str(path), encoding="utf-8")


@pytest.fixture
def root(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "ROOT", tmp_path.resolve())
    (tmp_path / "s02").mkdir()
    write_srt(tmp_path / "s02" / "a.tr.srt", ["Merhaba.", "Nasılsın?"])
    write_srt(tmp_path / "s02" / "a.en.srt", ["Hello.", "How are you?"])
    write_srt(tmp_path / "s02" / "b.tr.srt", ["Tamam."])  # no English yet
    return tmp_path


def test_episodes_lists_only_translated_pairs(root):
    assert api.episodes() == ["s02/a"]


def test_cues_pairs_turkish_with_english(root):
    c = api.cues("s02/a")
    assert c[1] == {"i": 1, "start": 2.0, "end": 3.5, "tr": "Nasılsın?", "en": "How are you?"}


def test_save_cue_edits_one_cue_and_keeps_original(root):
    api.save_cue(0, "s02/a", api.CueEdit(en="Hi."))
    assert [s.text for s in pysrt.open(str(root / "s02/a.en.srt"), encoding="utf-8")] == ["Hi.", "How are you?"]
    api.save_cue(1, "s02/a", api.CueEdit(en="How are you doing?"))
    assert [s.text for s in pysrt.open(str(root / "s02/a.en.srt.orig"), encoding="utf-8")] == ["Hello.", "How are you?"]
    assert not (root / "s02/a.en.srt.part").exists()


def test_rejects_paths_outside_root_and_missing_cues(root):
    for bad in ("../../etc/passwd", "s02/b", "nope"):
        with pytest.raises(HTTPException):
            api.cues(bad)
    with pytest.raises(HTTPException):
        api.save_cue(9, "s02/a", api.CueEdit(en="x"))
