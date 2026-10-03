# CLAUDE.md — Subtitle AI (Turkish Series)

AI-assisted pipeline that transcribes Turkish TV-series audio, translates it (default English), and produces timed subtitles (SRT/VTT/ASS) with a web review editor.

## Docs (read as needed)
- @docs/product-requirements.md — scope, requirements, quality targets
- @docs/architecture.md — pipeline stages, modules, data model, API
- @docs/design-system.md — UI tokens/components (frontend only)
- @docs/agent-guide.md — working agreements, playbooks, Turkish-specific rules

## Stack
**100% local, offline-capable.** Python 3.12 in Docker · faster-whisper · WhisperX · Silero VAD · pyannote · local LLM translator via Ollama/llama.cpp (pluggable; NLLB-200 as lightweight fallback) · FastAPI · SQLite · React + TypeScript + Vite.

## Hardware (target server)
Ryzen 5 5500 (12 threads) · 14 GB RAM · RTX 3070 **8 GB VRAM**. Run heavy models one at a time (Whisper int8_float16, then a Q4 7–9B LLM); unload between stages. Ollama is not installed yet.

## Layout
- `src/subai/pipeline/` — audio, transcribe, segmenter, srtio, runner (ingest → ASR → segment → SRT)
- `src/subai/translate/` — Translator protocol, prompts (versioned), glossary
- `web/` — review editor
- `eval/` — WER, readability, translation evals
- `docker/` — Dockerfile + compose.yaml; `scripts/subai` runs the CLI in Docker
- `docs/` — product, architecture, design, agent docs
- `requirements/` — base.txt (runtime), dev.txt (tests)
- `workspace/` — gitignored local I/O: `input/` (drop folder), `output/` (SRTs), `logs/`
- `/data` — **reserved for the media library** (host path, mounted read-only; never write there)

## Commands
```
./scripts/subai doctor
./scripts/subai run --input-dir "/data/media/.../{tvdb-383383}/Season 01" --recursive --diarize   # series glossary auto-detected
./scripts/subai glossary-fetch --series tvdb-383383 [--write]   # TMDB+TVDB refresh (needs API keys in docker/.env)
python3 eval/wer.py --ref REF.srt --hyp OUT.srt   # strict + content WER vs the human reference
./scripts/subai run -i /input/<file> | --input-dir /data/<folder> --batch
docker compose -f docker/compose.yaml run --rm --entrypoint pytest subai -q
(lint/type-check: planned)
```

## Rules
- **YAML trap:** bare `off`, `on`, `no`, `yes` in glossary YAML become booleans. Quote them (`"off"`); the loader rejects non-strings.
- Reference subtitles (human Turkish subs) are for scoring and learning punctuation conventions only; never copy their text into output.
- Pinned diarization stack: torch/torchaudio 2.5.1, huggingface_hub < 1.0, pyannote.audio 3.x (see requirements/diarize.txt). Do not bump casually.
- Stages are cached by input hash; bump a stage's `VERSION` when its output changes.
- Never edit a shipped prompt file in place; add a new version.
- Translate 1:1 per segment with surrounding context; cue merging/splitting is Format's job.
- Turkish text: Unicode NFC, locale-aware casing (İ/ı), suffix-aware glossary matching.
- Mock the translator and models in tests; no big model downloads in CI.
- **Docker is the supported runtime** (NVIDIA GPU via the NVIDIA Container Toolkit). `docker/Dockerfile` + `docker/compose.yaml`; media mounted read-only, output and model cache on volumes. Containers need no outbound network at runtime after models are cached.
- **Everything runs locally.** No cloud APIs, telemetry or network calls at runtime; no audio, video or text leaves the machine. Models are downloaded once, then used offline (`HF_HUB_OFFLINE=1`). Do not add a cloud provider.
- Local services bind 127.0.0.1 only. Never commit media, models or tokens (`HF_TOKEN` is only needed once to download gated pyannote weights).
- Use design tokens, never hard-coded colors; every control must be keyboard accessible.
- Keep changes minimal and tied to a PRD requirement (FR#). Ask before adding heavy dependencies or changing the data model.

## Status
Planning stage: docs only, no code yet. Next: M1 core pipeline (ingest → ASR → Turkish SRT).
