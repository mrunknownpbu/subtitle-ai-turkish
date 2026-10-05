"""Turkish SRT -> English SRT with opus-mt and deterministic Turkish handling. 1:1 per cue (merging/splitting is Format's job)."""
import logging
import re
import textwrap
from pathlib import Path

import pysrt

from subai.glossary import SeriesGlossary
from subai.output import write_subs_atomic
from subai.protect import (Entity, Glossary, bare_entity_translation, build_glossary, chunk_words, is_run_on,
                           missing_entities, phrase_key, protect, qc_flag, recover_dropped_entities, restore,
                           repair_corrupted_placeholders, split_dash_lines, split_sentences)

log = logging.getLogger(__name__)

_REPEAT = re.compile(r"\b(\w+)(?:[\s,.!?…-]+\1\b){2,}", re.I)


def collapse_repeats(text: str) -> str:
    """Whisper sometimes loops ('gel, gel, gel, ...'); the LLM loops on it too. Keep two repeats."""
    return _REPEAT.sub(lambda m: f"{m[1]} {m[1]}", text)


_HONORIFIC = ((re.compile(r"\b(Madam|Mrs?\.?|Ms\.?|Miss)(?=\W|$)"), "Hanım"), (re.compile(r"\b(Sir|Mr\.?)(?=\W|$)"), "Bey"))


_TITLE_FIRST = re.compile(r"\b(Bey|Hanım)\s+([A-ZÇĞİÖŞÜ]\w+)")


def fix_honorifics(src: str, en: str) -> str:
    """The model sometimes turns a Hanım/Bey cue into Madam/Mr despite the prompt."""
    for rx, tr in _HONORIFIC:
        if tr in src:
            en = rx.sub(tr, en)
    # the model sometimes writes the title first ("Bey Evren"); Turkish puts it after the name
    return _TITLE_FIRST.sub(r"\2 \1", en) if "Bey" in src or "Hanım" in src else en


def fix_gender(src: str, en: str, sg: SeriesGlossary | None) -> str:
    """Series terms with policy gender_fix: when the Turkish has the stem, swap the model's guessed word (grandson -> granddaughter)."""
    for t in (sg.terms if sg else []):
        if t.get("policy") == "gender_fix" and re.search(re.escape(t["tr"]), src, re.I):
            en = re.sub(rf"\b{t['wrong']}(s?)\b", lambda m: (t["en"].capitalize() if m[0][0].isupper() else t["en"]) + m[1], en, flags=re.I)
    return en


def wrap(text: str, width: int = 42) -> str:
    if "\n" in text or len(text) <= width or text.startswith("-"):
        return text
    return "\n".join(textwrap.wrap(text, width))


OPUS_MODEL = "Helsinki-NLP/opus-mt-tc-big-tr-en"


class Translator:
    """opus-mt-tc-big-tr-en on the GPU: fp16, beam 2, longest-first batches of 32, OOM halves the batch."""

    def __init__(self, name: str = OPUS_MODEL):
        import torch
        from transformers import MarianMTModel, MarianTokenizer

        if not torch.cuda.is_available():
            raise RuntimeError("opus-mt needs the GPU (nothing runs on the CPU here)")
        self.torch = torch
        self.tok = MarianTokenizer.from_pretrained(name)
        self.mt = MarianMTModel.from_pretrained(name, torch_dtype=torch.float16).cuda().eval()

    def __call__(self, texts: list[str], batch: int = 32) -> list[str]:
        order = sorted(range(len(texts)), key=lambda i: -len(texts[i]))
        out = [""] * len(texts)
        for a in range(0, len(order), batch):
            idx = order[a:a + batch]
            for i, t in zip(idx, self._generate([texts[i] for i in idx])):
                out[i] = t
        return out

    def _generate(self, xs: list[str]) -> list[str]:
        try:
            enc = self.tok(xs, return_tensors="pt", padding=True, truncation=True, max_length=512).to("cuda")
            with self.torch.inference_mode():
                ids = self.mt.generate(**enc, num_beams=2, max_new_tokens=256, no_repeat_ngram_size=4)
            return self.tok.batch_decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        except self.torch.cuda.OutOfMemoryError:
            if len(xs) == 1:
                raise
            self.torch.cuda.empty_cache()
            h = len(xs) // 2
            return self._generate(xs[:h]) + self._generate(xs[h:])

    def close(self) -> None:
        del self.mt
        self.torch.cuda.empty_cache()


def build_protection(sg: SeriesGlossary | None) -> Glossary:
    """Character names/aliases and glossary terms -> placeholder map. Character forms restore as written."""
    ents: list[Entity] = []
    for c in (sg.characters if sg else []):
        ents += [Entity(f, [f]) for f in [c.name, c.name.split()[0], *c.aliases]]
    for t in (sg.terms if sg else []):
        if t.get("policy") in ("translate", "keep"):
            ents.append(Entity(t["en"] if t["policy"] == "translate" else t["tr"], [t["tr"]]))
    return build_glossary(ents)


def build_phrase_map(lang: dict | None) -> dict[str, str]:
    return {phrase_key(e["tr"]): e["en"] for e in (lang or {}).get("phrase_map") or []}


def _phrase(s: str, en: str) -> str:
    tail = s.rstrip()[-1:]
    return en[:-1] + tail if tail in "!?" and en.endswith(".") else en


def translate_cues(texts: list[str], translate_fn, g: Glossary, phrase_map: dict[str, str]) -> list[str]:
    """One English text per Turkish text. Dash turns and sentences go through the model separately."""
    layout = []  # per cue: (is_dash, [[sentence index, ...] per unit])
    sents: list[str] = []
    for t in texts:
        lines = split_dash_lines(t)
        units = lines if lines else [" ".join(t.split())]
        idx = []
        for u in units:
            parts = split_sentences(u) or [u]
            idx.append(list(range(len(sents), len(sents) + len(parts))))
            sents += parts
        layout.append((bool(lines), idx))
    out = [""] * len(sents)
    todo: list[int] = []
    prot: dict[int, str] = {}
    for i, s in enumerate(sents):
        p = protect(s, g)
        if phrase_key(s) in phrase_map:
            out[i] = _phrase(s, phrase_map[phrase_key(s)])
        elif (b := bare_entity_translation(p, g)) is not None:
            out[i] = b
        elif not re.search(r"\w", s):
            out[i] = s
        else:
            todo.append(i)
            prot[i] = p
    raw = dict(zip(todo, translate_fn([prot[i] for i in todo]))) if todo else {}
    runon = [i for i in todo if is_run_on(sents[i])]
    if runon:  # a run-on goes in one call and gets garbled: retry in 6-word chunks and keep the better one
        chunks = [chunk_words(prot[i]) for i in runon]
        done = iter(translate_fn([c for cs in chunks for c in cs]))
        for i, cs in zip(runon, chunks):
            alt = " ".join(next(done) for _ in cs)
            a = restore(repair_corrupted_placeholders(raw[i], prot[i]), g)
            b = restore(repair_corrupted_placeholders(alt, prot[i]), g)
            ma, mb = (missing_entities(prot[i], x, g) for x in (a, b))
            if mb < ma or (mb == ma and len(a) < 0.25 * len(sents[i]) and len(b) > len(a)):
                raw[i] = alt
    for i in todo:
        t = restore(repair_corrupted_placeholders(raw[i], prot[i]), g)
        out[i] = recover_dropped_entities(prot[i], t, g)
    res = []
    for dash, units in layout:
        lines = [" ".join(out[i] for i in u).strip() for u in units]
        res.append("\n".join(f"- {x}" for x in lines) if dash else lines[0])
    return res


def translate_srt(src: Path, dst: Path, sg: SeriesGlossary | None, limit: int | None = None, *,
                     translate_fn=None, phrase_map: dict[str, str] | None = None) -> int:
    subs = pysrt.open(str(src), encoding="utf-8")
    if limit:
        subs = subs[:limit]
    tr = [collapse_repeats(s.text) for s in subs]
    mt = None if translate_fn else Translator()
    try:
        en = translate_cues(tr, translate_fn or mt, build_protection(sg), phrase_map or {})
    finally:
        if mt:
            mt.close()
    out = pysrt.SubRipFile()
    flagged = 0
    for i, s in enumerate(subs):
        text = fix_gender(s.text, fix_honorifics(s.text, en[i]), sg)
        text = text if any(l.startswith("-") for l in s.text.split("\n")) else text.removeprefix("- ")
        if why := qc_flag(tr[i], text):
            flagged += 1
            log.warning("cue %d flagged (%s): %r -> %r", i + 1, why, tr[i], text)
        out.append(pysrt.SubRipItem(index=i + 1, start=s.start, end=s.end, text=wrap(text)))
    log.info("translated %d cues, %d flagged by QC (advisory)", len(tr), flagged)
    write_subs_atomic(dst, out, allow_overwrite=True)  # a CLI run is an explicit request; the web layer decides KEEP/REPLACE
    return len(tr)
