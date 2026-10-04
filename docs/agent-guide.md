# Agent Guide — Subtitle AI (Turkish Series)

How AI coding agents should work in this repo. Read `CLAUDE.md` first, then this file, then the doc relevant to your task.

## 1. Read Order
1. `docs/product-requirements.md` — what and why
2. `docs/architecture.md` — pipeline, modules, data model
3. `docs/design-system.md` — only for `web/` work

## 2. Working Agreements
- Make the smallest change that satisfies the task. No drive-by refactors.
- Keep each pipeline stage pure-ish: input artifact in, output artifact out, no hidden global state.
- Never break stage caching. If a change alters stage output, bump that stage's version constant.
- Write tests first for formatter rules, glossary logic and JSON parsing (TDD).
- **Local only.** Never add cloud APIs, SDKs or telemetry. Do not download large models in tests; mock `Translator`; use tiny fixtures.
- Ask before: adding a heavy dependency, changing the data model, or changing default quality thresholds.

## 3. Task Playbooks

### Add / change a pipeline stage
1. Define input/output pydantic models in `models.py`.
2. Implement in `src/subai/pipeline/<stage>.py` with a `VERSION` constant.
3. Register in `runner.py` (order + cache key inputs).
4. Add unit test + a golden fixture if output is user-visible.
5. Update `architecture.md` diagram/section.

### Change the translation prompt
1. Copy the prompt to a new versioned file in `translate/prompts/` (e.g. `v3.md`); never edit shipped versions in place.
2. Run `eval/translation_eval.py` on the sample set and attach before/after numbers.
3. Keep JSON output schema backward compatible or bump parser version.

### Tune subtitle formatting rules
1. Change values in config defaults, not hard-coded constants.
2. Update golden SRT fixtures and explain each diff.
3. Verify readability compliance metric does not drop.

### Add a target language
1. Add a style guide in `translate/prompts/styles/<lang>.md` (formality, honorifics, punctuation, line length).
2. Add language code to config enum and UI selector.
3. Add 20-line eval set with reference translations.

### Frontend work
Use tokens from `design-system.md`; no hard-coded colors. Every interactive element needs a keyboard path and a focus style.

## 4. Turkish-Specific Knowledge
- Characters: ç Ç ğ Ğ ı I i İ ö Ö ş Ş ü Ü. Use Unicode NFC. **Case-fold carefully**: `"I".lower()` is wrong for Turkish — use locale-aware handling (`İ ↔ i`, `I ↔ ı`) in glossary matching.
- Turkish is agglutinative: glossary matching must handle suffixes (e.g. *Ayşe'nin*, *Ayşe'ye*). Match on stem + apostrophe suffix pattern; do not require whole-word equality.
- `o` is gender-neutral. Decide English *he/she* from the character sheet or speaker, and flag `gender_ambiguous` otherwise.
- Address forms: *abi, abla, hanım, bey, efendim, kardeşim*. Follow the series' honorific policy (keep / translate / drop).
- Idioms and oaths (*Allah Allah*, *Maşallah*, *geçmiş olsun*) are translated by function, not literally.
- Verb-final word order means a sentence's meaning can arrive late; translate with full-sentence context, never cue-by-cue.

## 5. Quality Gates (must pass before finishing)
```
docker compose -f docker/compose.yml run --rm --entrypoint pytest subai -q
# ruff / mypy: to be added to requirements/dev.txt
```
Frontend (if touched): `pnpm lint && pnpm typecheck && pnpm test`.

## 6. Do / Don't
**Do**
- Log stage name, input hash, duration, model versions.
- Fail loudly on malformed LLM JSON after bounded retries; surface in job status.
- Keep everything local; nothing (media, text, logs) may leave the machine.

**Don't**
- Don't commit media, models, `workspace/`, or tokens.
- Don't merge or split cues inside Translate; that belongs to Format.
- Don't "fix" ASR text silently in Translate — flag instead.
- Don't add features outside `product-requirements.md` without confirmation.

## 7. Definition of Done
- Behavior matches a requirement ID (FR#) from the PRD.
- Tests added and passing; quality gates green.
- Docs updated if architecture, config or prompts changed.
- Short summary of what changed and how it was verified.

## 8. Useful Commands
```
./scripts/subai run -i /input/ep01.mkv
./scripts/subai run -i /input/ep01.mkv --overwrite
python eval/wer.py workspace/eval/clean
```
