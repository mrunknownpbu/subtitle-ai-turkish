# Architecture — Subtitle AI (Turkish Series)

## 1. Overview
A staged, cache-backed pipeline plus a review web app. Each stage reads and writes versioned artifacts on disk, so any stage can be re-run independently.

```
video/audio
   │
   ▼
[1 Ingest]  ffmpeg → audio.wav (16 kHz mono)
   ▼
[2 Preprocess]  VAD (Silero) · optional vocal separation (Demucs)
   ▼
[3 ASR]  faster-whisper large-v3 (lang=tr)
   ▼
[4 Align]  forced alignment (WhisperX / wav2vec2-tr) → word timestamps
   ▼
[5 Diarize] (optional)  pyannote → speaker labels
   ▼
[6 Segment]  logical sentences / turns
   ▼
[7 Translate]  LLM + glossary + character sheet + context window
   ▼
[8 Format]  cue splitting, reading-speed, timing snap
   ▼
[9 Export]  SRT · VTT · ASS
   ▼
[Review UI]  edit → re-export
```

## 2. Tech Stack
| Layer | Choice | Reason |
|---|---|---|
| Language | Python 3.11 | ML ecosystem |
| ASR | faster-whisper (CTranslate2), Whisper large-v3 | Best open Turkish accuracy/speed |
| Alignment | WhisperX / wav2vec2 | Accurate word timings |
| VAD | Silero VAD | Cuts hallucination |
| Diarization | pyannote.audio | Speaker turns |
| Translation | Local LLM via Ollama / llama.cpp (candidates: Qwen, Gemma, Llama; quantized), NLLB-200 fallback; pluggable interface | Fully local, context-aware |
| API | FastAPI | Typed, async |
| Jobs | RQ + Redis (or in-process for CLI) | Simple resumable queue |
| DB | SQLite (SQLAlchemy) → Postgres optional | Local-first |
| Frontend | React + TypeScript + Vite, wavesurfer.js | Waveform editing |
| Packaging | uv (lockfile), native install, no Docker | Reproducible, direct GPU access |

## 3. Repository Layout
```
subtitle-ai-turkish/
├── CLAUDE.md  agent-guide.md  architecture.md  design-system.md  product-requirements.md
├── pyproject.toml
├── src/subai/
│   ├── cli.py                  # Typer entrypoint
│   ├── config.py               # pydantic-settings
│   ├── pipeline/
│   │   ├── runner.py           # stage orchestration + caching
│   │   ├── ingest.py  preprocess.py  asr.py  align.py  diarize.py
│   │   ├── segment.py  translate/  format.py  export.py
│   ├── translate/
│   │   ├── base.py             # Translator protocol
│   │   ├── ollama.py  llamacpp.py  nllb.py   # all local
│   │   ├── prompts/            # versioned prompt templates
│   │   └── glossary.py
│   ├── models.py               # pydantic domain models
│   ├── storage.py              # artifact store, content hashing
│   └── api/                    # FastAPI routers
├── web/                        # React review editor
├── tests/                      # unit, golden, e2e
├── data/                       # gitignored: media, artifacts
└── eval/                       # WER, readability, translation eval sets
```

## 4. Domain Model
- **Series** (id, name, target_langs, glossary, characters, style_guide)
- **Episode** (id, series_id, source_path, status, config_snapshot)
- **Segment** (id, episode_id, start, end, text_tr, words[], speaker, confidence)
- **Cue** (id, episode_id, start, end, lines[], text_target, flags[], source_segment_ids[])
- **GlossaryEntry** (term_tr, translation, note, scope)
- **Character** (name, aliases, gender, register, relationships)
- **Job / StageRun** (stage, input_hash, output_path, model_versions, duration)

## 5. Artifact Store & Caching
`data/episodes/<id>/<stage>/<input_hash>/…`
Cache key = hash(stage input + stage config + model/prompt version). Re-running with unchanged inputs is a no-op. Editing glossary invalidates only translate → export.

## 6. Key Stage Designs

### ASR
- 30 s chunks driven by VAD boundaries, `condition_on_previous_text=False` (limits looping), beam 5, temperature fallback.
- Filters: drop segments with high `no_speech_prob`, repeated n-grams, or compression ratio > 2.4.
- Optional initial prompt with series character names to bias spelling.

### Segmentation
Merge words into sentence/turn units using punctuation and pauses (>400 ms) and speaker changes.

### Translation
- Batches of ~20 segments with ±5 lines context on each side and a running scene summary.
- Prompt injects: glossary, character sheet, target language style guide, honorific policy.
- Structured JSON in/out (id → translation + flags); schema validated (use Ollama JSON-schema constrained output), retries on mismatch. Smaller local models need shorter batches (~10 segments).
- Model is a config value; chosen by running `eval/translation_eval.py` over candidates.
- GPU memory (RTX 3070, 8 GB): ASR and LLM run strictly sequentially; unload each model after its stage. Use `compute_type=int8_float16` for Whisper and Q4 7–9B LLMs. Keep context ≤ 4–8k tokens. Diarization/alignment also run alone.
- System RAM is 14 GB: avoid loading several models in CPU memory at once; set Ollama `OLLAMA_MAX_LOADED_MODELS=1`.
- Temperature low (≤0.3). Line count preserved 1:1 with segments; merging happens in Format.
- Flags: `idiom`, `low_confidence`, `name_unknown`, `gender_ambiguous`, `too_long`.

### Formatting
Constraint solver per segment: split on clause boundaries to satisfy ≤42 chars/line, ≤2 lines, CPS ≤17; extend end time into gaps up to limit; enforce min gap. Rules live in config, covered by golden tests.

## 7. API (FastAPI, local)
```
POST /series            POST /series/{id}/episodes
GET  /episodes/{id}     POST /episodes/{id}/run?from=stage
GET  /episodes/{id}/cues        PATCH /cues/{id}
POST /episodes/{id}/export?fmt=srt|vtt|ass
GET/PUT /series/{id}/glossary
WS   /episodes/{id}/progress
```

## 8. Frontend
Single-page editor: media player + waveform, virtualized cue table, side-by-side TR/target, flagged-queue filter, shortcut-driven. See `design-system.md`.

## 9. Deployment
- Dev: `uv run subai …` and `pnpm dev`.
- GPU: native install, no Docker. Python deps via `uv sync` (pinned CUDA-enabled wheels), system ffmpeg, Ollama as a native service. Models cached under `~/.cache` / `data/models`.
- No inbound network exposure by default (binds 127.0.0.1).

## 10. Security & Privacy
- Fully local: no outbound network at runtime. Models fetched once, then run with `HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1`; Ollama on 127.0.0.1.
- `HF_TOKEN` is needed only once to download gated pyannote weights; never stored in the repo.
- Optional egress check in tests: fail if the pipeline opens non-loopback connections.
- Uploaded file paths validated; ffmpeg invoked with argument lists, never shell strings.

## 11. Testing & Evaluation
- Unit: formatter rules, glossary application, JSON schema handling.
- Golden: fixed 2-minute clips → expected SRT diff tolerance.
- Eval harness (`eval/`): WER on labeled Turkish clips, readability compliance, LLM-judge + human sample for translation.
- Translator mocked in CI; real local-model runs behind a `slow` marker.

## 12. Decisions Log
| # | Decision | Rationale |
|---|---|---|
| 1 | Pipeline of cached stages | Cheap iteration on translation without re-running ASR |
| 2 | Pluggable Translator protocol, local backends only | Swap local models as better ones appear |
| 5 | No cloud dependencies | Privacy and offline use are requirements |
| 3 | 1:1 segment translation, merge later | Keeps timing traceable |
| 4 | SQLite + files | Local-first simplicity |
