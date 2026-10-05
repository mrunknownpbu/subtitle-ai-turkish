# Product Requirements — Subtitle AI (Turkish Series)

## 1. Purpose
Generate broadcast-quality subtitles for Turkish TV series: transcribe Turkish speech, translate to a target language (default English; Malay and Indonesian next), and export timed subtitle files. A human reviewer polishes the result in a lightweight editor.

## 2. Users
| User | Need |
|---|---|
| Fan subtitler / small distributor | Fast first draft, minimal manual cleanup |
| Reviewer / editor | Fix errors quickly with audio-synced editing |
| Developer (operator) | Run the pipeline via CLI/API on batches of episodes |

## 3. Problem
- Turkish dialogue is fast, overlapping, emotional, with idioms, honorifics (abi, abla, hanım, bey), dialect and slang.
- Generic machine translation loses tone, gender (Turkish has no grammatical gender: *o*), formality and character voice.
- Raw ASR timestamps drift and produce unreadable subtitle blocks.

## 4. Goals
1. Episode (45 min) → reviewed-ready subtitles in under 15 min on a single GPU.
2. Turkish transcription WER ≤ 12% on clean dialogue; ≤ 20% with background music.
3. Translation judged acceptable (no meaning error) on ≥ 90% of lines by reviewer sampling.
4. Subtitles pass readability rules (see §7) with ≥ 98% compliance.
5. Consistent character names, terms and register across an entire series.

## 5. Non-Goals (v1)
- Live/real-time subtitling.
- Dubbing or voice synthesis.
- Burning subtitles into video (export files only; optional ffmpeg helper later).
- Supporting source languages other than Turkish.

## 6. Functional Requirements

### FR1 Ingest
- Accept video (mp4, mkv, avi) or audio (wav, mp3, m4a) upload or local path.
- Extract mono 16 kHz audio with ffmpeg; preserve original untouched.
- Projects group episodes into a **Series** with shared glossary.

### FR2 Transcription
- Turkish ASR with word-level timestamps (Whisper large-v3 via faster-whisper + forced alignment).
- Voice activity detection to skip music/silence and reduce hallucinations.
- Optional music/vocal separation for noisy scenes.
- Optional speaker diarization (who spoke) to support dialogue dashes and gender/register hints.
- Output: Turkish transcript segments with confidence scores.

### FR3 Translation
- Local machine translation (opus-mt-tc-big-tr-en, in-process on the GPU) wrapped in deterministic Turkish handling: sentence and dash-turn splitting, glossary name protection, fixed phrases.
- Uses series glossary and character sheet (names, relationships, speech register, gender).
- Preserves proper nouns, handles idioms by meaning, keeps honorific policy configurable (keep / translate / drop).
- Targets: English (v1), Malay, Indonesian (v1.1); pluggable.
- Flags low-confidence lines for review.

### FR4 Subtitle Formatting
- Re-segment into subtitle cues obeying readability rules (§7).
- Export SRT, WebVTT, ASS. Dual-language export (Turkish + target) optional.

### FR5 Review Editor
- Web UI: video/audio player, waveform, cue list, side-by-side Turkish/target.
- Edit text and timing, split/merge cues, jump to flagged lines.
- Keyboard-first workflow; autosave; version history per episode.
- Glossary edits can be re-applied to selected cues.

### FR6 Series Memory
- Glossary (term → fixed translation), character list, tone notes persisted per series.
- Reviewer corrections suggest new glossary entries.

### FR7 Batch & Ops
- CLI: `subai run <file> --target en`; batch folder mode.
- Job queue with progress, retry, resumable stages (cache intermediate outputs).
- Cost/time report per episode.

## 7. Subtitle Quality Rules (defaults, configurable)
- Max 2 lines per cue, ≤ 42 characters per line.
- Reading speed ≤ 17 characters/second (≤ 20 for fast dialogue).
- Duration 1.0 s – 7.0 s; min gap 2 frames (≈ 80 ms).
- Break at clause/phrase boundaries; no orphan words on a line.
- Dialogue by two speakers in one cue: each line starts with "- ".
- No cues over music-only sections unless lyrics flagged by user.

## 8. Non-Functional Requirements
- **Performance:** ≥ 4× real-time on a 12 GB+ GPU; CPU fallback supported (slow).
- **Local-only:** all processing (ASR, translation, UI) runs on the user's machine. No cloud APIs, accounts or telemetry; works fully offline after one-time model download. No media or text leaves the machine.
- **Target hardware (current server):** AMD Ryzen 5 5500 (6C/12T), 14 GB RAM, NVIDIA RTX 3070 **8 GB VRAM**, ~700 GB free disk. Design for this as the baseline: Whisper large-v3 (int8_float16, ~3–4 GB) and a 7–9B quantized (Q4) LLM (~5–6 GB) run **sequentially**, never together. 12–14B+ LLMs need partial CPU offload (slow, RAM-limited) and are optional. Bigger GPUs only improve speed/quality. CPU-only mode supported with smaller models, much slower.
- **Reproducibility:** each run stores model versions, prompts and config.
- **Reliability:** every stage idempotent and cached by content hash.
- **Accessibility:** editor meets WCAG 2.2 AA; full keyboard operation.
- **Legal:** the tool processes media the user has rights to; docs state this.

## 9. Success Metrics
| Metric | Target |
|---|---|
| WER (Turkish, clean) | ≤ 12% |
| Reviewer edit rate (cues changed) | ≤ 30% |
| Readability compliance | ≥ 98% |
| Time per 45-min episode (auto stages) | ≤ 15 min |
| Glossary consistency (named terms) | ≥ 99% |

## 10. Milestones
1. **M1 – Core pipeline (CLI):** ingest → ASR → SRT in Turkish.
2. **M2 – Translation:** context-aware LLM translation to English + formatter rules.
3. **M3 – Series memory:** glossary, character sheet, flagging.
4. **M4 – Review editor (web).**
5. **M5 – Diarization, music separation, Malay/Indonesian targets, batch mode.**

## 11. Risks & Mitigations
| Risk | Mitigation |
|---|---|
| Whisper hallucination in silence/music | VAD, no-speech thresholds, repetition filter |
| Wrong pronoun gender in translation | Diarization + character sheet + reviewer flags |
| Timing drift | Forced alignment, snap cues to word timestamps |
| LLM inconsistency across episodes | Glossary injection, series memory, low temperature |
| Local LLM weaker than frontier models on Turkish idioms | Benchmark candidates (Qwen, Gemma, Llama, NLLB) on an eval set; glossary + character sheet; flagging; reviewer pass |
| 8 GB VRAM / 14 GB RAM limits | Run stages sequentially, unload models between stages, Q4 quantized 7–9B LLMs, small batches, avoid running diarization and LLM concurrently; speed target (§4) to be re-validated on this hardware |
| Copyright misuse | Terms/README notice; no content distribution features |

## 12. Open Questions
- Which 7–9B local LLM is the default for 8 GB VRAM (decide via eval on Turkish→English sample set)? Is a GPU upgrade planned?
- Target languages beyond English priority order?
- Is a hosted multi-user version needed, or local single-user only?
