"""Language + series glossaries (YAML) and helpers.

Layout (mounted at /glossary in Docker, ./glossary in the repo):
  glossary/language/<lang>.yaml
  glossary/series/<slug>-tvdb-<id>/{series,characters,terms}.yaml
"""
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

log = logging.getLogger(__name__)

DEFAULT_DIR = Path(os.environ.get("SUBAI_GLOSSARY_DIR", "/glossary"))
TVDB_RE = re.compile(r"tvdb-(\d+)")
GENDERS = {"male", "female", "unknown"}
ROLES = ("lead", "main", "recurring", "guest")


class GlossaryError(Exception):
    pass


@dataclass
class Character:
    name: str
    aliases: list[str]
    actor: str
    gender: str
    role: str
    episodes: int = 0
    note: str = ""


@dataclass
class SeriesGlossary:
    id: str
    title: dict
    meta: dict
    characters: list[Character] = field(default_factory=list)
    terms: list[dict] = field(default_factory=list)
    corrections: list[dict] = field(default_factory=list)  # explicit {heard, correct} ASR fixes


_WORD = "A-Za-z0-9_çğıöşüÇĞİÖŞÜ"


def apply_corrections(text: str, corrections: list[dict]) -> tuple[str, int]:
    """Replace whole words (case-sensitive; suffixes after an apostrophe stay: Sarkan'ın -> Serkan'ın)."""
    n = 0
    for c in corrections:
        text, k = re.subn(rf"(?<![{_WORD}]){re.escape(c['heard'])}(?![{_WORD}])", c["correct"], text)
        n += k
    return text, n


def is_hallucination(text: str, patterns: list[str]) -> bool:
    """True if the whole cue is a known Whisper phantom phrase (e.g. 'Altyazı M.K.')."""
    t = re.sub(rf"[^{_WORD}]+", " ", text.replace("İ", "i").replace("I", "ı")).lower().strip()
    return any(re.fullmatch(p, t) for p in patterns)


def detect_series_id(path: Path | str) -> str | None:
    """Find 'tvdb-<id>' anywhere in a media path (Plex/Sonarr style '{tvdb-383383}')."""
    m = TVDB_RE.search(str(path))
    return f"tvdb-{m.group(1)}" if m else None


def find_series_dir(glossary_dir: Path, series_id: str) -> Path | None:
    for d in sorted((glossary_dir / "series").glob(f"*{series_id}")):
        if d.is_dir():
            return d
    return None


def _read(path: Path) -> dict:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise GlossaryError(f"cannot read {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise GlossaryError(f"{path} must contain a mapping")
    return data


def load_series(glossary_dir: Path, series_id: str) -> SeriesGlossary | None:
    d = find_series_dir(glossary_dir, series_id)
    if d is None:
        return None
    meta = _read(d / "series.yaml")
    if meta.get("id") != series_id:
        raise GlossaryError(f"{d}/series.yaml id {meta.get('id')!r} != folder id {series_id!r}")
    chars = []
    for i, c in enumerate(_read(d / "characters.yaml").get("characters", [])):
        for key in ("name", "actor", "gender", "role"):
            if not c.get(key):
                raise GlossaryError(f"characters[{i}] missing '{key}'")
        if c["gender"] not in GENDERS:
            raise GlossaryError(f"characters[{i}] {c['name']}: bad gender {c['gender']!r}")
        if c["role"] not in ROLES:
            raise GlossaryError(f"characters[{i}] {c['name']}: bad role {c['role']!r}")
        chars.append(Character(c["name"], list(c.get("aliases", [])), c["actor"], c["gender"],
                               c["role"], int(c.get("episodes", 0)), c.get("note", "")))
    tdata = _read(d / "terms.yaml")
    corrections = tdata.get("asr_corrections", [])
    for i, c in enumerate(corrections):
        if not c.get("heard") or not c.get("correct"):
            raise GlossaryError(f"asr_corrections[{i}] needs 'heard' and 'correct'")
    return SeriesGlossary(series_id, meta.get("title", {}), meta, chars, tdata.get("terms", []), corrections)


def load_language(glossary_dir: Path, lang: str = "tr") -> dict:
    path = glossary_dir / "language" / f"{lang}.yaml"
    data = _read(path)
    for key, items in (("tone.exclamation_interjections", (data.get("tone") or {}).get("exclamation_interjections") or []),
                       ("asr_hallucinations", data.get("asr_hallucinations") or [])):
        bad = [x for x in items if not isinstance(x, str)]
        if bad:
            raise GlossaryError(f"{path}: {key} must be strings, got {bad!r} "
                                "(YAML reads bare off/on/no/yes as booleans: put them in quotes)")
    for section in ("address_terms", "formulae", "idioms_and_phrases"):
        for i, e in enumerate(data.get(section, [])):
            if not e.get("tr") or not e.get("en"):
                raise GlossaryError(f"{path}: {section}[{i}] needs 'tr' and 'en'")
    return data


def asr_prompt(series: SeriesGlossary, max_chars: int = 300) -> str:
    """Comma-separated names for Whisper's initial_prompt, most important characters first."""
    rank = {r: i for i, r in enumerate(ROLES)}
    out: list[str] = []
    for c in sorted(series.characters, key=lambda c: (rank[c.role], -c.episodes)):
        forms = [c.name.split()[0]] + [a for a in c.aliases if " " not in a and a != c.name.split()[0]]
        for f in forms:
            if f not in out and len(", ".join(out + [f])) <= max_chars:
                out.append(f)
    return ", ".join(out)
