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
| Language | Python 3.12 (Ubuntu 24.04 image) | ML ecosystem |
| ASR | faster-whisper (CTranslate2), Whisper large-v3 | Best open Turkish accuracy/speed |
| Alignment | WhisperX / wav2vec2 | Accurate word timings |
| VAD | Silero VAD | Cuts hallucination |
| Diarization | pyannote.audio | Speaker turns |
| Translation | Local LLM via Ollama / llama.cpp (candidates: Qwen, Gemma, Llama; quantized), NLLB-200 fallback; pluggable interface | Fully local, context-aware |
| API | FastAPI | Typed, async |
| Jobs | RQ + Redis (or in-process for CLI) | Simple resumable queue |
| DB | SQLite (SQLAlchemy) → Postgres optional | Local-first |
| Frontend | React + TypeScript + Vite, wavesurfer.js | Waveform editing |
| Packaging | Docker (CUDA 12 + cuDNN 9 base image), compose; Python deps pinned in image | Reproducible, no host dependency drift |

## 3. Repository Layout
```
subtitle-ai-turkish/
├── README.md  LICENSE  CLAUDE.md
├── docs/                       # product-requirements, architecture, design-system, agent-guide
├── docker/                     # Dockerfile, compose.yaml
├── scripts/subai               # runs the CLI inside Docker
├── requirements/               # base.txt (runtime), dev.txt (tests)
├── src/subai/
│   ├── __main__.py  cli.py     # Typer entrypoint (run, doctor, download-model)
│   ├── api.py                  # FastAPI wrapper (planned)
│   ├── logs.py  doctor.py  models.py
│   ├── pipeline/               # implemented: audio, transcribe, segmenter, srtio, runner
│   │                           # planned: align, diarize, format, export
│   └── translate/              # planned: base, ollama, llamacpp, nllb, prompts/, glossary
├── tests/                      # unit, golden, e2e
├── web/                        # React review editor (planned)
├── eval/                       # WER, readability, translation eval sets (planned)
└── workspace/                  # gitignored local I/O: input/ output/ logs/ artifacts/
                                # (media library lives at /data on the host, read-only)
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
`workspace/artifacts/<id>/<stage>/<input_hash>/…`
Cache key = hash(stage input + stage config + model/prompt version). Re-running with unchanged inputs is a no-op. Editing glossary invalidates only translate → export.

## 6. Key Stage Designs

### ASR
- 30 s chunks driven by VAD boundaries, `condition_on_previous_text=False` (limits looping), beam 5, temperature fallback.
- VAD (Silero via faster-whisper): `threshold 0.3, min_silence 300 ms, speech_pad 400 ms` (library default 0.5 clipped short utterances and caused errors like *persepsiyona* for *resepsiyona*).
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
- Run: `./scripts/subai <command>` (wraps `docker compose -f docker/compose.yaml run --rm subai`).
- Docker + NVIDIA Container Toolkit. `docker/compose.yaml` services: `subai` (CLI/batch, run on demand) and `api` (FastAPI on 127.0.0.1). Mounts: media library read-only (`/data`), `./workspace` for input/output/logs, a named volume for the model cache. `ollama` (LLM translation, bound to 127.0.0.1, volume `subai-ollama`, one model loaded) is reached by `subai` over the compose network, never the internet.
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
| 6 | Whisper `large-v3`, not `large-v3-turbo` | Turbo is ~2.5x faster but clearly worse on Turkish (garbled phrases) in a 3-min drama-clip comparison |
| 7 | Sensitive VAD, beam 5, no `hotwords`, `initial_prompt` optional | Greedy decoding was worse; `hotwords` dropped whole lines; beam 8 gave no measurable gain; names via `--initial-prompt` |
| 8 | Read extracted WAV ourselves, not via PyAV | faster-whisper 1.2.1 passes `metadata_errors` to `av.open`, which PyAV 19 rejects |
| 9 | Repair stray words (group of <=3 words >3 s away from the rest of its Whisper segment) | After long silence Whisper stamps a segment's first word far too early ("Bu" 108 s before "arada Serkanlar nerede?") |
| 10 | Hallucination blocklist in the language glossary (`asr_hallucinations`) | "Altyazı M.K." appeared at the end of 4 of 5 episodes; whole-cue match only |
| 11 | Cue split pause 0.3 s (was 0.7 s) | Boundary F1 vs human subtitle breaks 0.704 -> 0.729; cues/episode 2,107 vs the human 2,105 |
| 12 | Tone step = continuation dots + interjection "!" only | Learned on S01E01-04, held-out S01E05: precision 91% / 92%. Commas rejected (precision <= 52% at every threshold) |
| 13 | Decoding frozen at large-v3, int8_float16, beam 5, sensitive VAD | 8-variant sweep (prompts, fp16, beam 8, no-speech) all within 14.6-15.1% content WER = noise |
| 14 | Diarization = pyannote 3.1 in a subprocess; pins torch/torchaudio 2.5.1, huggingface_hub < 1.0 | pyannote 3.x breaks on torchaudio >= 2.9 and hub 1.x; subprocess isolates PyTorch's CUDA libs from CTranslate2 and frees VRAM |
| 15 | Speaker turn splits a cue only at a sentence end or after a >= 0.25 s pause | Diarizers misplace boundaries mid-sentence ("- Ben / - tanıştırayım."). Measured: detects 25-35% of human-marked exchanges with ~3% false splits |

| 16 | Vocal separation (Demucs htdemucs) rejected | Whisper large-v3 is already robust to the show's music: content WER 14.63% original vs 14.80% vocals-only vs 15.82% blended |
| 17 | QLoRA fine-tune on the series, labels = our transcript + reference corrections and punctuation | Held-out WER 14.47% -> 13.49%; commas 3.2 -> 7.9 per 100 words (reference 7.5); timing unchanged. See docs/finetuning.md |
| 18 | Translation = opus-mt draft + Qwen3 8B edit (Ollama), 1:1 per cue, honorific post-fix | S01E05 chrF vs human English: Qwen alone 48.5, opus-mt 51.4, opus-mt + Qwen edit 51.7; Gemma 12B runs partly on CPU at 8 GB. See docs/translation.md |

## 13. Evaluation (eval/wer.py)
Human Turkish hearing-impaired subtitles for S01E01-05 are used as reference, never as input.
Scores: *strict* WER and *content* WER (spelling variants unified, fillers ignored, circumflex dropped).
Baseline on 52,091 reference words: strict 16.5%, content 14.6%. Much of the remainder is style: the
reference is condensed and written-standard ("vallahi"), the speech is colloquial ("valla").
Do not tune toward the reference's spelling: the goal is natural spoken tone, with the reference used for
content accuracy, cue-break agreement and punctuation conventions.

