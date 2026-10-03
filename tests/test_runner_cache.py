from subai.models import Word
from subai.pipeline import runner


class FakeTranscriber:
    def __init__(self):
        self.calls = 0
        self.model = "fake"

    def settings(self):
        return {"model": self.model}

    def transcribe(self, wav, duration=0.0):
        self.calls += 1
        return [Word("Merhaba", 0.0, 0.5), Word("Sarkan", 0.5, 1.0)]


def _fake_extract(src, dst, track=None):
    dst.write_bytes(b"x" * 2000)
    return 10.0


def _setup(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "extract_audio", _fake_extract)
    src = tmp_path / "ep.mkv"
    src.write_bytes(b"media")
    return src, tmp_path / "out" / "ep.tr.srt", FakeTranscriber()


def test_cache_skips_asr_and_reapplies_corrections(tmp_path, monkeypatch):
    src, out, tr = _setup(tmp_path, monkeypatch)
    runner.process_file(src, out, tr, None)
    assert tr.calls == 1 and "Sarkan" in out.read_text(encoding="utf-8")

    runner.process_file(src, out, tr, None, corrections=[{"heard": "Sarkan", "correct": "Serkan"}])
    assert tr.calls == 1  # served from the cache
    assert "Serkan" in out.read_text(encoding="utf-8")

    runner.process_file(src, out, tr, None, force_asr=True)
    assert tr.calls == 2


def test_cache_invalidated_when_settings_change(tmp_path, monkeypatch):
    src, out, tr = _setup(tmp_path, monkeypatch)
    runner.process_file(src, out, tr, None)
    tr.model = "other-model"
    runner.process_file(src, out, tr, None)
    assert tr.calls == 2


def test_corrupt_cache_is_ignored(tmp_path, monkeypatch):
    src, out, tr = _setup(tmp_path, monkeypatch)
    runner.process_file(src, out, tr, None)
    (out.parent / ".subai" / "ep.words.json").write_text("{not json", encoding="utf-8")
    runner.process_file(src, out, tr, None)
    assert tr.calls == 2
