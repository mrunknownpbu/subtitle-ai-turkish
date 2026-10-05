import pysrt
import pytest

from subai.output import OutputSafetyError, write_srt_atomic, write_subs_atomic


def test_write_is_atomic_keeps_existing_and_leaves_no_temp(tmp_path):
    f = tmp_path / "a.en.srt"
    assert write_srt_atomic(f, "one", allow_overwrite=False) is True
    assert write_srt_atomic(f, "two", allow_overwrite=False) is False and f.read_text() == "one"  # KEEP
    assert write_srt_atomic(f, "two", allow_overwrite=True) is True and f.read_text() == "two"
    assert oct(f.stat().st_mode & 0o777) == "0o644"
    assert [p.name for p in tmp_path.iterdir()] == ["a.en.srt"]


def test_stale_temp_files_do_not_block_a_write(tmp_path):
    (tmp_path / ".a.srt.part").write_text("stale")
    (tmp_path / ".a.srt.tmp-old").write_text("stale")
    write_srt_atomic(tmp_path / "a.srt", "ok", allow_overwrite=True)
    assert (tmp_path / "a.srt").read_text() == "ok"


def test_reference_subtitles_are_never_written(tmp_path):
    for name in ("x.en.hi.srt", "x.en.forced.srt", "x.en.sdh.srt"):
        with pytest.raises(OutputSafetyError):
            write_srt_atomic(tmp_path / name, "no", allow_overwrite=True)
        assert not (tmp_path / name).exists()


def test_write_subs_renders_srt(tmp_path):
    subs = pysrt.SubRipFile(items=[pysrt.SubRipItem(1, start=1000, end=2500, text="hi\nthere")])
    write_subs_atomic(tmp_path / "a.srt", subs, allow_overwrite=True)
    back = pysrt.open(str(tmp_path / "a.srt"), encoding="utf-8")
    assert back[0].text == "hi\nthere" and back[0].end.ordinal == 2500
