"""Deterministic Turkish handling around the MT model (ported from the sibling project subtitle-ai).

Pure text functions, no torch. Names and fixed terms are swapped for opaque placeholders (Xaa, Xab, ...)
before translation and restored after; short utterances bypass the model; multi-speaker and multi-sentence
cues are split so the model never has to keep them apart.
"""
import re
from dataclasses import dataclass

_ASCII_WORD = "[A-Za-z0-9_]"  # ASCII boundary: "Serkan'ın" splits at the apostrophe, the stem protects cleanly
_PLACEHOLDER = r"X[a-z]{2}"


def _bounded(pattern: str) -> str:
    return rf"(?<!{_ASCII_WORD}){pattern}(?!{_ASCII_WORD})"


@dataclass
class Entity:
    canonical: str       # restored into the English output
    forms: list[str]     # Turkish spellings to protect


Glossary = dict[str, tuple[str, str]]  # form (original spelling) -> (placeholder, canonical)


def _placeholder(k: int) -> str:
    return "X" + chr(97 + k // 26) + chr(97 + k % 26)


def build_glossary(entities: list[Entity]) -> Glossary:
    """Keyed on the ORIGINAL spelling: casefold() turns Turkish İ into two codepoints, so it only dedups."""
    g: Glossary = {}
    seen: set[str] = set()
    for e in entities:
        for form in e.forms:
            if len(form) > 2 and form.casefold() not in seen:
                seen.add(form.casefold())
                g[form] = (_placeholder(len(g)), e.canonical)
    return g


def protect(text: str, g: Glossary) -> str:
    for form, (ph, _) in sorted(g.items(), key=lambda kv: -len(kv[0])):  # "Serkan Bolat" before "Serkan"
        text = re.sub(_bounded(re.escape(form)), ph, text, flags=re.I)
    return text


def restore(text: str, g: Glossary) -> str:
    for ph, canonical in g.values():
        text = re.sub(_bounded(re.escape(ph)), lambda _m, c=canonical: c, text, flags=re.I)
    return text


def _one_edit_away(a: str, b: str) -> bool:
    return len(a) == len(b) and sum(x != y for x, y in zip(a, b)) <= 1


def repair_corrupted_placeholders(candidate: str, source_protected: str) -> str:
    """Marian sometimes returns 'Xax' for 'Xac'. Fix only an unambiguous near miss of a placeholder in this sentence."""
    active = set(re.findall(_bounded(_PLACEHOLDER), source_protected))
    if not active:
        return candidate

    def fix(m: re.Match) -> str:
        t = m[0]
        near = [p for p in active if _one_edit_away(t, p)]
        return t if t in active or len(near) != 1 else near[0]

    return re.sub(_bounded(_PLACEHOLDER), fix, candidate)


def bare_entity_translation(protected: str, g: Glossary) -> str | None:
    """Text that is only placeholders + punctuation ('Cenk.') is restored without the model, which would invent words."""
    residual, hit = protected, False
    for ph, _ in g.values():
        residual, n = re.subn(_bounded(re.escape(ph)), "", residual, flags=re.I)
        hit = hit or bool(n)
    return restore(protected, g) if hit and not re.search(r"\w", residual) else None


def _count(text: str, canonical: str) -> int:
    return len(re.findall(_bounded(re.escape(canonical)), text, re.I))


def entity_report(source_protected: str, target: str, g: Glossary) -> dict[str, tuple[int, int]]:
    """canonical -> (occurrences in source, occurrences in target) for entities present in the source."""
    out: dict[str, tuple[int, int]] = {}
    for ph, canonical in g.values():
        n = len(re.findall(_bounded(re.escape(ph)), source_protected, re.I))
        if n and canonical not in out:
            out[canonical] = (n, _count(target, canonical))
    return out


def recover_dropped_entities(source_protected: str, restored: str, g: Glossary) -> str:
    """Prepend a canonical name the model dropped. Never calls the model, so it cannot invent content."""
    tail = source_protected.rstrip()[-1:]
    trailing = tail if tail in ".!?" else ""
    for canonical, (src, tgt) in entity_report(source_protected, restored, g).items():
        if src > tgt:
            ins = " ".join(f"{canonical}{trailing}" for _ in range(src - tgt))
            restored = f"{ins} {restored}".strip()
    return restored


def missing_entities(source_protected: str, restored: str, g: Glossary) -> int:
    return sum(max(s - t, 0) for s, t in entity_report(source_protected, restored, g).values())


def split_dash_lines(text: str) -> list[str] | None:
    """'- a\\n- b' -> ['a', 'b'] when every line is a dash turn (2+ lines), else None."""
    lines = text.split("\n")
    if len(lines) < 2:
        return None
    m = [re.match(r"^-\s*(.*)$", ln) for ln in lines]
    return [x[1] for x in m] if all(m) else None


_SENT_END = re.compile(r"[.!?…]+(?:\s+|$)")


def split_sentences(text: str) -> list[str] | None:
    """Two or more sentences -> list (punctuation kept); a single sentence -> None."""
    pieces, i = [], 0
    for m in _SENT_END.finditer(text):
        if text[i:m.end()].strip():
            pieces.append(text[i:m.end()].strip())
        i = m.end()
    if text[i:].strip():
        pieces.append(text[i:].strip())
    return pieces if len(pieces) > 1 else None


def is_run_on(text: str) -> bool:
    """8+ words and no sentence punctuation: Marian garbles these in one call."""
    return not re.search(r"[.!?…]", text) and len(text.split()) >= 8


def chunk_words(text: str, n: int = 6) -> list[str]:
    w = text.split()
    return [" ".join(w[i:i + n]) + "." for i in range(0, len(w), n)]


def phrase_key(text: str) -> str:
    """Whole-utterance lookup key; Turkish-aware lowering (İ->i, I->ı), trailing .!? ignored."""
    return text.strip().rstrip(".!?").replace("İ", "i").replace("I", "ı").lower()


def qc_flag(src: str, out: str) -> str | None:
    """Advisory only: a reason the English looks wrong, or None."""
    if not out.strip():
        return "empty"
    if re.search(_bounded(_PLACEHOLDER), out):
        return "placeholder leaked"
    if sum(map(out.count, "([")) != sum(map(out.count, ")]")):
        return "unbalanced bracket"
    ratio = len(out) / max(len(src), 1)
    return "too short" if ratio < 0.25 else "too long" if ratio > 3.5 else None
