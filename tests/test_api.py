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


def test_jobs_queue_run_in_order_and_a_queued_job_can_be_cancelled(tmp_path, monkeypatch):
    import sys
    import time

    monkeypatch.setattr(api, "ROOT", tmp_path)
    monkeypatch.setattr(api, "RETIMES", {})
    (tmp_path / "uploads").mkdir()
    sleep = lambda s: [sys.executable, "-c", f"import time; time.sleep({s})", "x"]
    cleaned = []
    a = api._start_job([("slow", sleep(0.6))], "uploads/a", "a.mkv")["id"]
    b = api._start_job([("never", sleep(0))], "uploads/b", "b.mkv", cleanup=lambda: cleaned.append("b"))["id"]
    c = api._start_job([("last", sleep(0))], "uploads/c", "c.mkv")["id"]
    assert api.RETIMES[b]["state"] == "queued"  # waits behind a, not rejected
    api.cancel_job(b)
    assert api.RETIMES[b]["state"] == "cancelled"
    for _ in range(100):
        if api.RETIMES[c]["state"] == "done":
            break
        time.sleep(0.1)
    assert [api.RETIMES[i]["state"] for i in (a, b, c)] == ["done", "cancelled", "done"]
    assert cleaned == ["b"]  # the worker skipped it and still cleaned up
    assert api.RETIMES[a]["ended"] <= api.RETIMES[c]["began"]  # strictly one at a time


def test_upload_keeps_an_existing_file_unless_replace(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "ROOT", tmp_path)
    one = b"1\n00:00:01,000 --> 00:00:02,000\nmerhaba\n\n"
    two = b"1\n00:00:01,000 --> 00:00:02,000\ngule gule\n\n"
    assert api._store_srt("ep", "tr", one) == "uploads/ep"
    with pytest.raises(HTTPException) as e:
        api._store_srt("ep", "tr", two)
    assert e.value.status_code == 409 and "merhaba" in (tmp_path / "uploads/ep.tr.srt").read_text()
    api._store_srt("ep", "tr", two, replace=True)
    assert "gule gule" in (tmp_path / "uploads/ep.tr.srt").read_text()


def test_retime_keeps_the_existing_episode_and_names_the_copy_retimed(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "ROOT", tmp_path)
    (tmp_path / "uploads").mkdir()
    assert api._retimed_stem("ep", False) == "ep"
    (tmp_path / "uploads/ep.tr.srt").write_text("x")
    assert api._retimed_stem("ep", False) == "ep.retimed" and api._retimed_stem("ep", True) == "ep"


def test_job_step_is_skipped_as_keep_when_its_output_exists(tmp_path, monkeypatch):
    import sys

    monkeypatch.setattr(api, "ROOT", tmp_path)
    monkeypatch.setattr(api, "RETIMES", {})
    out = tmp_path / "out.srt"
    out.write_text("mine")
    writes = [sys.executable, "-c", f"open({str(out)!r}, 'w').write('machine')", "x"]
    job = api.RETIMES["j"] = {"state": "running", "log": []}
    api._run_job(job, [("Translating", writes, out)], lambda: None, jid="j")
    assert out.read_text() == "mine" and job["state"] == "done" and "KEEP: out.srt already exists" in job["log"]
    api._run_job(job, [("Translating", writes, None)], lambda: None, jid="j")  # replace requested: no keep path
    assert out.read_text() == "machine"


def test_editor_save_is_refused_while_a_job_works_on_the_episode(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "ROOT", tmp_path)
    (tmp_path / "uploads").mkdir()
    for lang in ("tr", "en"):
        (tmp_path / f"uploads/ep.{lang}.srt").write_text("1\n00:00:01,000 --> 00:00:02,000\nx\n\n")
    monkeypatch.setattr(api, "RETIMES", {"j": {"state": "running", "ep": "uploads/ep", "log": []}})
    with pytest.raises(HTTPException) as e:
        api.save_cue(0, "uploads/ep", api.CueEdit(en="hello"))
    assert e.value.status_code == 409
    api.RETIMES["j"]["state"] = "done"
    assert api.save_cue(0, "uploads/ep", api.CueEdit(en="hello"))["en"] == "hello"


def test_save_cue_mirrors_to_the_video_folder(root, tmp_path, monkeypatch):
    media = tmp_path / "media"
    media.mkdir()
    video = media / "ep.mkv"
    video.write_bytes(b"x")
    (media / "ep.en.hi.srt").write_text("human reference", encoding="utf-8")
    monkeypatch.setattr(api, "_source", lambda ep: (video, False))
    assert api.save_cue(0, "s02/a", api.CueEdit(en="Hi."))["media_error"] is None
    assert [s.text for s in pysrt.open(str(media / "ep.en.srt"), encoding="utf-8")] == ["Hi.", "How are you?"]
    assert (media / "ep.en.hi.srt").read_text(encoding="utf-8") == "human reference"
    monkeypatch.setattr(api, "_source", lambda ep: (None, True))  # no video: the edit still lands in /output
    assert "no video" in api.save_cue(1, "s02/a", api.CueEdit(en="Fine."))["media_error"]
    assert pysrt.open(str(root / "s02/a.en.srt"), encoding="utf-8")[1].text == "Fine."


def test_publish_copies_only_what_the_job_wrote(root, tmp_path, monkeypatch):
    import time

    media = tmp_path / "media"
    media.mkdir()
    (media / "a.mkv").write_bytes(b"x")
    monkeypatch.setattr(api, "MEDIA", media)
    write_srt(root / "s02" / "a.tr.srt", ["Yeni."])
    tr = root / "s02" / "a.tr.srt"
    tr.write_bytes(tr.read_bytes().replace(b"\n", b"\r\n"))  # retime writes CRLF; the copy must keep it
    write_srt(root / "s02" / "a.en.srt", ["New."])
    (media / "a.en.srt").write_text("kept", encoding="utf-8")
    old = time.time() - 100  # the English file predates the job: a step kept it
    import os
    os.utime(root / "s02" / "a.en.srt", (old, old))
    api.RETIMES["j"] = {"began": time.time() - 10, "log": []}
    api._publish("j", "s02/a", "a.mkv")
    assert [s.text for s in pysrt.open(str(media / "a.tr.srt"), encoding="utf-8")] == ["Yeni."]
    assert (media / "a.tr.srt").read_bytes() == (root / "s02" / "a.tr.srt").read_bytes()
    assert (media / "a.en.srt").read_text(encoding="utf-8") == "kept"
    write_srt(root / "s02" / "a.retimed.tr.srt", ["R."])
    api._publish("j", "s02/a.retimed", "a.mkv")
    assert not (media / "a.retimed.tr.srt").exists()


def test_finish_review_deletes_the_episodes_files_only(root, tmp_path, monkeypatch):
    media = tmp_path / "media"
    media.mkdir()
    video = media / "a.mkv"
    video.write_bytes(b"x")
    (media / "a.en.srt").write_text("media copy", encoding="utf-8")
    d = root / "s02"
    for name in ("a.en.srt.orig", "a.video", "a.web.mp4", "a.peaks.json", ".subai/a.words.json"):
        (d / name).parent.mkdir(exist_ok=True)
        (d / name).write_text("x")
    (d / ".subai" / "b.words.json").write_text("other episode")
    monkeypatch.setattr(api, "_source", lambda ep: (video, False))
    assert "s02/a" in api.episodes()
    api.RETIMES["j"] = {"ep": "s02/a", "state": "running", "log": []}
    with pytest.raises(HTTPException) as e:
        api.finish_review("s02/a")
    assert e.value.status_code == 409 and (d / "a.tr.srt").exists()
    api.RETIMES["j"]["state"] = "done"
    assert api.finish_review("s02/a")["removed"] == 7
    assert sorted(p.name for p in d.rglob("*") if p.is_file()) == ["b.tr.srt", "b.words.json"]
    assert (media / "a.en.srt").read_text(encoding="utf-8") == "media copy" and video.exists()
    (root / ".backup" / "s02").mkdir(parents=True)
    write_srt(root / ".backup/s02/z.tr.srt", ["x"]); write_srt(root / ".backup/s02/z.en.srt", ["x"])
    assert not any(e.startswith(".backup") for e in api.episodes())
