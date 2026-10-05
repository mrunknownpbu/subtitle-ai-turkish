"""The one place subtitle files are written (same method as the sibling project subtitle-ai).

Temp file with a unique name in the target directory -> fsync -> atomic rename, so a reader never sees a partial file and
a crash never leaves a stale fixed-name temp that blocks later writes. `allow_overwrite=False` keeps an existing file
and returns False (a normal outcome, not an error). Human reference subtitles are never written.
"""
import io
import os
import tempfile
from pathlib import Path

import pysrt

# Never written, by any code path, even with allow_overwrite. They are reference subtitles for scoring only.
PROTECTED_SUFFIXES = (".en.hi.srt", ".en.forced.srt", ".en.sdh.srt")


class OutputSafetyError(Exception):
    pass


def is_protected(path: str | Path) -> bool:
    return Path(path).name.endswith(PROTECTED_SUFFIXES)


def write_srt_atomic(path: str | Path, content: str, *, allow_overwrite: bool) -> bool:
    """True if written, False if the file exists and allow_overwrite is False."""
    target = Path(path)
    if is_protected(target):
        raise OutputSafetyError(f"refusing to write a protected reference subtitle: {target.name}")
    if target.exists() and not allow_overwrite:
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{target.name}.tmp-", dir=target.parent)
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            os.fchmod(fh.fileno(), 0o644)
            fh.write(content)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, target)
    finally:
        tmp.unlink(missing_ok=True)
    return True


def write_subs_atomic(path: str | Path, subs: pysrt.SubRipFile, *, allow_overwrite: bool) -> bool:
    buf = io.StringIO()
    subs.write_into(buf)
    return write_srt_atomic(path, buf.getvalue(), allow_overwrite=allow_overwrite)
