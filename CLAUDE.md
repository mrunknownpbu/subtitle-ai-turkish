# CLAUDE.md — Subtitle AI (Turkish Series)

AI-assisted pipeline that transcribes Turkish TV-series audio, translates it (default English), and produces timed subtitles (SRT/VTT/ASS) with a web review editor.

## Docs (read as needed)
- @docs/product-requirements.md — scope, requirements, quality targets
- @docs/architecture.md — pipeline stages, modules, data model, API
- @docs/design-system.md — UI tokens/components (frontend only)
- @docs/translation.md — Turkish→English translation: design, measurements, limits
- @docs/finetuning.md — QLoRA fine-tuning of Whisper on the series (train/prepare.py, train/finetune.py)
- @docs/agent-guide.md — working agreements, playbooks, Turkish-specific rules

## Stack
**100% local, offline-capable.** Python 3.12 in Docker · faster-whisper · WhisperX · Silero VAD · pyannote · local LLM translator via Ollama/llama.cpp (pluggable; NLLB-200 as lightweight fallback) · FastAPI · SQLite · React + TypeScript + Vite.

## Hardware (target server)
Ryzen 5 5500 (12 threads) · 14 GB RAM · RTX 3070 **8 GB VRAM**. Run heavy models one at a time (Whisper int8_float16, then a Q4 7–9B LLM); unload between stages. Ollama runs as a compose service (stop it before Whisper jobs).

## Environments
- **Testing site:** this directory, `/opt/projects/subtitle-ai-turkish/`. Build, test and experiment here.
- **Production:** `/opt/docker/`. Only deploy here when the user asks.
  - `compose/` — compose files, always named `compose.yml`; shared `.env` at `compose/.env`
  - `appdata/` — persistent storage and application data
  - `script/` — custom scripts

## Layout
- `src/subai/pipeline/` — audio, transcribe, segmenter, srtio, runner (ingest → ASR → segment → SRT)
- `src/subai/translate.py` — Turkish SRT → English: opus-mt draft, Qwen3 edit via Ollama (see docs/translation.md)
- `web/` — review editor
- `eval/` — WER, readability, translation evals
- `docker/` — Dockerfile + compose.yml; `scripts/subai` runs the CLI in Docker
- `docs/` — product, architecture, design, agent docs
- `requirements/` — base.txt (runtime), dev.txt (tests)
- `train/` — dataset builder + QLoRA script; `workspace/models/` holds fine-tuned CTranslate2 models (mounted at /ft)
- `workspace/` — gitignored local I/O: `input/` (drop folder), `output/` (SRTs), `logs/`
- `/data` — **reserved for the media library** (host path, mounted read-only; never write there)

## Commands
```
./scripts/subai doctor
./scripts/subai run --input-dir "/data/media/.../{tvdb-383383}/Season 01" --recursive --diarize   # series glossary auto-detected
./scripts/subai glossary-fetch --series tvdb-383383 [--write]   # TMDB+TVDB refresh (needs API keys in docker/.env)
python3 eval/wer.py --ref REF.srt --hyp OUT.srt   # strict + content WER vs the human reference
docker compose -f docker/compose.yml up -d ollama   # once; then:
./scripts/subai translate -i "/output/<name>.tr.srt" --series tvdb-383383   # -> <name>.en.srt
python3 eval/chrf.py --ref REF.en.hi.srt --hyp OUT.en.srt   # English chrF vs the human subtitle
./scripts/subai run -i /input/<file> | --input-dir /data/<folder> --batch
docker compose -f docker/compose.yml run --rm --entrypoint pytest subai -q
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
- **Docker is the supported runtime** (NVIDIA GPU via the NVIDIA Container Toolkit). `docker/Dockerfile` + `docker/compose.yml`; media mounted read-only, output and model cache on volumes. Containers need no outbound network at runtime after models are cached.
- **Everything runs locally.** No cloud APIs, telemetry or network calls at runtime; no audio, video or text leaves the machine. Models are downloaded once, then used offline (`HF_HUB_OFFLINE=1`). Do not add a cloud provider.
- Local services bind 127.0.0.1 only. Never commit media, models or tokens (`HF_TOKEN` is only needed once to download gated pyannote weights).
- Use design tokens, never hard-coded colors; every control must be keyboard accessible.
- Keep changes minimal and tied to a PRD requirement (FR#). Ask before adding heavy dependencies or changing the data model.

## Status
Transcription (v2 fine-tuned Whisper, tone, diarization, glossary) and Turkish→English translation (opus-mt draft + Qwen3 edit) work per episode. Next: Format stage (cue merge/split, reading speed), export formats, review UI.
