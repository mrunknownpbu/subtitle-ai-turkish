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


SRT = "1\n00:00:01,000 --> 00:00:02,000\nMerhaba.\n\n2\n00:00:03,000 --> 00:00:04,000\nNasılsın?\n"


def test_upload_creates_pair_with_blank_other_language(root):
    ep = api._store_srt("My Show S01E01.srt", "tr", SRT.encode("utf-8"))
    assert ep == "uploads/My Show S01E01"
    assert api.episodes() == ["s02/a", ep]
    c = api.cues(ep)
    assert [(x["tr"], x["en"], x["start"]) for x in c] == [("Merhaba.", "", 1.0), ("Nasılsın?", "", 3.0)]
    api._store_srt("My Show S01E01.srt", "en", SRT.replace("Merhaba.", "Hello.").encode())  # real English replaces the blank
    assert api.cues(ep)[0]["en"] == "Hello."


def test_upload_reads_legacy_turkish_and_rejects_junk(root):
    ep = api._store_srt("old.srt", "tr", SRT.encode("cp1254"))
    assert api.cues(ep)[1]["tr"] == "Nasılsın?"
    for name, lang, data in (("x", "de", SRT.encode()), ("x", "tr", b"not a subtitle"), ("..", "tr", SRT.encode())):
        with pytest.raises(HTTPException):
            api._store_srt(name, lang, data)


def test_video_choice_and_browse_stay_inside_library(root, tmp_path, monkeypatch):
    lib = tmp_path / "lib"
    (lib / "show").mkdir(parents=True)
    (lib / "show" / "e1.mkv").write_bytes(b"x")
    (lib / "show" / "notes.txt").write_text("x")
    monkeypatch.setattr(api, "MEDIA", lib)
    api._index.cache_clear()
    assert api.browse("show") == {"path": "show", "dirs": [], "files": ["e1.mkv"]}
    assert api.video_choice("s02/a") == {"file": "", "auto": True}
    assert api.set_video_choice("s02/a", api.VideoChoice(file="show/e1.mkv")) == {"file": "show/e1.mkv", "auto": False}
    assert api.set_video_choice("s02/a", api.VideoChoice()) == {"file": "", "auto": True}
    for bad in ("../x.mkv", "show/notes.txt", "show/missing.mkv"):
        with pytest.raises(HTTPException):
            api.set_video_choice("s02/a", api.VideoChoice(file=bad))
    with pytest.raises(HTTPException):
        api.browse("..")


def test_rejects_paths_outside_root_and_missing_cues(root):
    for bad in ("../../etc/passwd", "s02/b", "nope"):
        with pytest.raises(HTTPException):
            api.cues(bad)
    with pytest.raises(HTTPException):
        api.save_cue(9, "s02/a", api.CueEdit(en="x"))


def test_browse_can_list_subtitles_instead_of_videos(tmp_path, monkeypatch):
    (tmp_path / "show").mkdir()
    for n in ("e1.mkv", "e1.tr.srt", "e1.en.srt", "notes.txt", ".hidden.srt"):
        (tmp_path / "show" / n).write_text("x")
    monkeypatch.setattr(api, "MEDIA", tmp_path)
    assert api.browse("show", "srt")["files"] == ["e1.en.srt", "e1.tr.srt"]
    assert api.browse("show")["files"] == ["e1.mkv"]


def test_process_validates_before_starting_anything(tmp_path, monkeypatch):
    import asyncio

    (tmp_path / "a.mkv").write_bytes(b"x")
    (tmp_path / "a.txt").write_text("x")
    monkeypatch.setattr(api, "MEDIA", tmp_path)

    class Req:  # stands in for the request body
        async def body(self):
            return b""

    def call(**kw):
        with pytest.raises(HTTPException) as e:
            asyncio.run(api.process(Req(), **kw))
        return e.value.status_code

    assert call(video="a.txt") == 400                                  # not a video
    assert call(video="../a.mkv") == 400                               # outside the library
    assert call(video="a.mkv", transcribe=False, translate=False) == 400
    assert call(video="a.mkv", transcribe=False, translate=True) == 400  # translate-only needs a Turkish SRT
    assert call(video="a.mkv", model="not-a-model") == 400
    assert call(video="a.mkv", transcribe=False, translate=False, retime=False) == 400  # no step chosen
    assert call(video="a.mkv", transcribe=True, translate=False, retime=True) == 400    # retime replaces transcription
    assert call(video="a.mkv", transcribe=False, translate=True, retime=True) == 400    # retime needs an SRT


def test_job_history_survives_restart_and_clears(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "ROOT", tmp_path)
    monkeypatch.setattr(api, "RETIMES", {})
    api.RETIMES["a"] = {"state": "running", "log": [], "ep": "uploads/x", "started": 1.0}
    api.RETIMES["b"] = {"state": "done", "log": [], "ep": "uploads/y", "started": 2.0}
    for i in ("a", "b"):
        api._save_job(i)
    api.RETIMES.clear()
    api._load_jobs()  # a fresh server process
    assert api.RETIMES["a"]["state"] == "failed" and "restart" in api.RETIMES["a"]["log"][-1]
    assert api.RETIMES["b"]["state"] == "done"
    assert [j["id"] for j in api.jobs()] == ["b", "a"]  # newest first
    assert api.jobs_clear() == {"cleared": 2}
    assert api.jobs() == [] and not list((tmp_path / ".jobs").glob("*.json"))
