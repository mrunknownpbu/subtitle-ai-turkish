"""Turkish SRT -> English SRT via a local Ollama model. 1:1 per cue (merging/splitting is Format's job)."""
import json
import logging
import os
import re
import textwrap
import urllib.request
from pathlib import Path

import pysrt

from subai.glossary import SeriesGlossary

log = logging.getLogger(__name__)

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://ollama:11434")
PROMPT_VERSION = "v1"
BATCH = 8
CONTEXT = 3
SCHEMA = {
    "type": "object",
    "properties": {"translations": {"type": "array", "items": {
        "type": "object",
        "properties": {"id": {"type": "integer"}, "en": {"type": "string"}},
        "required": ["id", "en"]}}},
    "required": ["translations"],
}


def system_prompt(sg: SeriesGlossary | None, target: str = "English", drafts: bool = False) -> str:
    p = [f"You are a professional subtitle translator from Turkish to {target} for a TV series. "
         "Translate each numbered cue into natural, concise spoken English that keeps the speaker's tone "
         "(teasing, anger, warmth). Rules:",
         "- Return exactly one translation per input id; never merge, split, skip or renumber cues.",
         "- Cues are consecutive lines of dialogue and often continue each other (Turkish puts the verb last): "
         "use 'previous' and 'next' for context but translate only 'cues'.",
         "- Do not add dashes or quotes. Only if a source line starts with '- ' (a second speaker), keep it.",
         "- Always translate every cue into English, including song lyrics; never copy Turkish text through.",
         "- Turkish 'o' has no gender: pick he/she/they from context and the character list.",
         "- Translate idioms and exclamations (Allah Allah, Maşallah, geçmiş olsun) by meaning, not literally.",
         "- Keep proper names; never turn 'Bey'/'Hanım' into Mr/Mrs/Madam/Sir, keep them after the name (Ayfer Hanım); translate 'abi', 'abla' only when no name is given.",
         "- Suffix -cığım/-ciğim/-cım (Eda'cığım) is affectionate: write 'dear Eda' or just 'Eda', never copy the suffix.",
         "- 'Hala,' / 'Hala' used to address someone means 'Auntie' (not 'still'); likewise Teyze, Abla, Abi.",
         "- Common idioms: kafayı yemek = go crazy; bayılmak = to love; inşallah = God willing / hopefully; yeğen = niece (girl) or nephew (boy).",
         "- Keep ellipses and '!' / '?' where the source has them. Output only the translation, no notes."]
    if drafts:
        p.append("Each cue has a machine 'draft' translation. Keep it where it is correct and natural; "
                 "fix wrong meaning, gender, names, idioms and missing or invented content from the Turkish.")
    if sg:
        t = sg.title
        p.append(f"Series: {t.get('original')} ({t.get('english_release')}). {' '.join(str(sg.meta.get('synopsis', '')).split())}")
        p.append("Characters: " + "; ".join(f"{c.name} ({c.gender})" + (f" - {c.note}" if c.note else "")
                                            for c in sg.characters if c.role in ("lead", "main")))
        terms = [f"{x['tr']} = {x['en']}" for x in sg.terms if x.get("policy") == "translate"]
        keep = [x["tr"] for x in sg.terms if x.get("policy") == "keep"]
        if terms:
            p.append("Fixed translations: " + "; ".join(terms))
        if keep:
            p.append("Never translate: " + ", ".join(keep))
    return "\n".join(p)


def chat(model: str, system: str, user: str) -> str:
    body = {"model": model, "stream": False, "think": False, "format": SCHEMA,
            "options": {"temperature": 0, "num_ctx": 6144},
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
    req = urllib.request.Request(f"{OLLAMA_URL}/api/chat", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)["message"]["content"]


def _ask(model, system, ids, tr, en, tries=3, drafts=None):
    """Translate cue indices `ids`; returns {idx: text}. Falls back to single cues, then raises."""
    payload = {"previous": [{"tr": tr[i], "en": en[i]} for i in range(max(0, ids[0] - CONTEXT), ids[0]) if i in en],
               "cues": [{"id": i, "tr": tr[i], **({"draft": drafts[i]} if drafts else {})} for i in ids],
               "next": [tr[i] for i in range(ids[-1] + 1, min(len(tr), ids[-1] + 1 + CONTEXT))]}
    for _ in range(tries):
        try:
            got = {int(t["id"]): t["en"].strip() for t in json.loads(chat(model, system, json.dumps(payload, ensure_ascii=False)))["translations"]}
        except (ValueError, KeyError, TypeError) as exc:
            log.warning("bad model JSON (%s), retrying", exc)
            continue
        if set(got) == set(ids) and all(got.values()):
            return got
        log.warning("id mismatch %s vs %s, retrying", sorted(got), ids)
    if len(ids) > 1:
        return {k: v for i in ids for k, v in _ask(model, system, [i], tr, en, tries, drafts).items()}
    raise RuntimeError(f"model failed to translate cue {ids[0]}: {tr[ids[0]]!r}")


_HONORIFIC = ((re.compile(r"\b(Madam|Mrs?\.?|Ms\.?|Miss)(?=\W|$)"), "Hanım"), (re.compile(r"\b(Sir|Mr\.?)(?=\W|$)"), "Bey"))


_TITLE_FIRST = re.compile(r"\b(Bey|Hanım)\s+([A-ZÇĞİÖŞÜ]\w+)")


def fix_honorifics(src: str, en: str) -> str:
    """The model sometimes turns a Hanım/Bey cue into Madam/Mr despite the prompt."""
    for rx, tr in _HONORIFIC:
        if tr in src:
            en = rx.sub(tr, en)
    # the model sometimes writes the title first ("Bey Evren"); Turkish puts it after the name
    return _TITLE_FIRST.sub(r"\2 \1", en) if "Bey" in src or "Hanım" in src else en


def wrap(text: str, width: int = 42) -> str:
    if "\n" in text or len(text) <= width or text.startswith("-"):
        return text
    return "\n".join(textwrap.wrap(text, width))


OPUS_MODEL = "Helsinki-NLP/opus-mt-tc-big-tr-en"


def opus_draft(tr: list[str]) -> list[str]:
    """One cue -> one English draft with opus-mt (0.4 GB, GPU, ~25 s per episode); frees the GPU after."""
    import torch
    from transformers import MarianMTModel, MarianTokenizer

    tok = MarianTokenizer.from_pretrained(OPUS_MODEL)
    mt = MarianMTModel.from_pretrained(OPUS_MODEL, torch_dtype=torch.float16).cuda().eval()
    out: list[str] = []
    for i in range(0, len(tr), 16):
        enc = tok([" ".join(t.split()) for t in tr[i:i + 16]], return_tensors="pt", padding=True).to("cuda")
        out += tok.batch_decode(mt.generate(**enc, num_beams=2, max_new_tokens=128), skip_special_tokens=True)
    del mt
    torch.cuda.empty_cache()
    return out


def load_drafts(draft: str | Path | None, tr: list[str]) -> list[str] | None:
    """draft: None/'none' = no draft, 'opus' = opus-mt, otherwise an English .srt with the same cues."""
    if draft in (None, "none"):
        return None
    if draft == "opus":
        return opus_draft(tr)
    drafts = [d.text.replace("\n", " ") for d in pysrt.open(str(draft), encoding="utf-8")]
    if len(drafts) < len(tr):
        raise ValueError(f"draft has {len(drafts)} cues, source has {len(tr)}")
    return drafts[:len(tr)]


def translate_srt(src: Path, dst: Path, model: str, sg: SeriesGlossary | None, limit: int | None = None,
                  draft: str | Path | None = "opus") -> int:
    subs = pysrt.open(str(src), encoding="utf-8")
    if limit:
        subs = subs[:limit]
    tr = [s.text for s in subs]
    en: dict[int, str] = {}
    drafts = load_drafts(draft, tr)
    system = system_prompt(sg, drafts=bool(drafts))
    for a in range(0, len(tr), BATCH):
        en.update(_ask(model, system, list(range(a, min(a + BATCH, len(tr)))), tr, en, drafts=drafts))
        log.info("translated %d/%d", len(en), len(tr))
    out = pysrt.SubRipFile()
    for i, s in enumerate(subs):
        text = fix_honorifics(s.text, en[i])
        text = text if any(l.startswith("-") for l in s.text.split("\n")) else text.removeprefix("- ")
        out.append(pysrt.SubRipItem(index=i + 1, start=s.start, end=s.end, text=wrap(text)))
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".part")
    out.save(str(tmp), encoding="utf-8")
    tmp.replace(dst)
    return len(tr)
