# CLAUDE.md — Subtitle AI (Turkish Series)

AI-assisted pipeline that transcribes Turkish TV-series audio, translates it (default English), and produces timed subtitles (SRT/VTT/ASS) with a web review editor.

## Docs (read as needed)
- @product-requirements.md — scope, requirements, quality targets
- @architecture.md — pipeline stages, modules, data model, API
- @design-system.md — UI tokens/components (frontend only)
- @agent-guide.md — working agreements, playbooks, Turkish-specific rules

## Stack
**100% local, offline-capable.** Python 3.11 (uv) · faster-whisper · WhisperX · Silero VAD · pyannote · local LLM translator via Ollama/llama.cpp (pluggable; NLLB-200 as lightweight fallback) · FastAPI · SQLite · React + TypeScript + Vite.

## Hardware (target server)
Ryzen 5 5500 (12 threads) · 14 GB RAM · RTX 3070 **8 GB VRAM**. Run heavy models one at a time (Whisper int8_float16, then a Q4 7–9B LLM); unload between stages. Ollama is not installed yet.

## Layout
- `src/subai/pipeline/` — stages: ingest → preprocess → asr → align → diarize → segment → translate → format → export
- `src/subai/translate/` — Translator protocol, prompts (versioned), glossary
- `web/` — review editor
- `eval/` — WER, readability, translation evals
- `data/` — gitignored media and cached artifacts

## Commands
```
uv sync
uv run subai run <media> --series <name> --target en
uv run pytest -q
uv run ruff check . && uv run mypy src
pnpm --dir web dev
```

## Rules
- Stages are cached by input hash; bump a stage's `VERSION` when its output changes.
- Never edit a shipped prompt file in place; add a new version.
- Translate 1:1 per segment with surrounding context; cue merging/splitting is Format's job.
- Turkish text: Unicode NFC, locale-aware casing (İ/ı), suffix-aware glossary matching.
- Mock the translator and models in tests; no big model downloads in CI.
- **No Docker.** Native install only (uv, system ffmpeg, native Ollama); don't add Dockerfiles or compose files.
- **Everything runs locally.** No cloud APIs, telemetry or network calls at runtime; no audio, video or text leaves the machine. Models are downloaded once, then used offline (`HF_HUB_OFFLINE=1`). Do not add a cloud provider.
- Local services bind 127.0.0.1 only. Never commit media, models or tokens (`HF_TOKEN` is only needed once to download gated pyannote weights).
- Use design tokens, never hard-coded colors; every control must be keyboard accessible.
- Keep changes minimal and tied to a PRD requirement (FR#). Ask before adding heavy dependencies or changing the data model.

## Status
Planning stage: docs only, no code yet. Next: M1 core pipeline (ingest → ASR → Turkish SRT).
