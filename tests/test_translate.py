import json

import pysrt
import pytest

from subai import translate as T


def fake_chat(reply):
    """Stand-in for T.chat; reply(payload) -> {id: en}. Records the payloads it saw."""
    seen = []

    def chat(model, system, user):
        payload = json.loads(user)
        seen.append(payload)
        return json.dumps({"translations": [{"id": i, "en": t} for i, t in reply(payload).items()]})

    chat.seen = seen
    return chat


def write_srt(path, texts):
    subs = pysrt.SubRipFile(items=[
        pysrt.SubRipItem(i + 1, start=i * 2000, end=i * 2000 + 1500, text=t) for i, t in enumerate(texts)])
    subs.save(str(path), encoding="utf-8")


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


def test_ask_retries_then_falls_back_to_single_cues(monkeypatch):
    calls = []

    def reply(payload):
        ids = [c["id"] for c in payload["cues"]]
        calls.append(ids)
        return {ids[0]: "x"} if len(ids) > 1 else {ids[0]: f"en{ids[0]}"}  # a batch never returns every id

    monkeypatch.setattr(T, "chat", fake_chat(reply))
    assert T._ask("m", "sys", [0, 1], ["a", "b"], {}, tries=2) == {0: "en0", 1: "en1"}
    assert calls == [[0, 1], [0, 1], [0], [1]]


def test_ask_raises_when_a_single_cue_never_translates(monkeypatch):
    monkeypatch.setattr(T, "chat", fake_chat(lambda p: {}))
    with pytest.raises(RuntimeError, match="cue 0"):
        T._ask("m", "sys", [0], ["a"], {}, tries=2)


def test_translate_srt_dashes_honorifics_and_drafts(monkeypatch, tmp_path):
    src, dst = tmp_path / "a.tr.srt", tmp_path / "a.en.srt"
    write_srt(src, ["Ayfer Hanım.", "- Gel.\n- Gelmem.", "Tamam."])
    answers = {0: "Madam Ayfer.", 1: "- Come.\n- No.", 2: "- Okay."}
    chat = fake_chat(lambda p: {c["id"]: answers[c["id"]] for c in p["cues"]})
    monkeypatch.setattr(T, "chat", chat)
    monkeypatch.setattr(T, "opus_draft", lambda tr: [f"draft {i}" for i in range(len(tr))])
    assert T.translate_srt(src, dst, "m", None) == 3
    out = [s.text for s in pysrt.open(str(dst), encoding="utf-8")]
    assert out == ["Ayfer Hanım.", "- Come.\n- No.", "Okay."]  # a dash survives only for a two-speaker cue
    assert [c["draft"] for c in chat.seen[0]["cues"]] == ["draft 0", "draft 1", "draft 2"]
    assert not (tmp_path / "a.en.srt.part").exists()


def test_load_drafts_modes(tmp_path):
    assert T.load_drafts("none", ["a"]) is None and T.load_drafts(None, ["a"]) is None
    d = tmp_path / "d.srt"
    write_srt(d, ["one", "two\nlines"])
    assert T.load_drafts(d, ["a", "b"]) == ["one", "two lines"]
    with pytest.raises(ValueError):
        T.load_drafts(d, ["a", "b", "c"])


def test_system_prompt_mentions_drafts_only_when_given():
    assert "draft" not in T.system_prompt(None)
    assert "draft" in T.system_prompt(None, drafts=True)
