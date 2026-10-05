"""SRT writing (atomic, UTF-8)."""
from pathlib import Path

import pysrt

from subai.models import Cue
from subai.output import write_subs_atomic


def read_srt(data: bytes) -> pysrt.SubRipFile:
    """Parse SRT bytes: UTF-8 (BOM allowed), else legacy Turkish cp1254."""
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = data.decode("cp1254", errors="replace")
    return pysrt.from_string(text)


def write_srt(cues: list[Cue], path: Path) -> None:
    subs = pysrt.SubRipFile()
    for i, c in enumerate(cues, 1):
        subs.append(
            pysrt.SubRipItem(
                index=i,
                start=pysrt.SubRipTime.from_ordinal(int(c.start * 1000)),
                end=pysrt.SubRipTime.from_ordinal(int(c.end * 1000)),
                text=c.text,
            )
        )
    write_subs_atomic(path, subs, allow_overwrite=True)
