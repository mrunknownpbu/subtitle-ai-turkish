# Subtitle AI — Turkish Series

Fully local, AI-assisted subtitle generator for Turkish TV series. It transcribes Turkish speech, translates it (English first; Malay and Indonesian planned), and exports timed subtitles as SRT, WebVTT and ASS.

> **Status: M1 working.** Local Turkish transcription to SRT runs end to end in Docker (about 21x realtime on an RTX 3070), with series/language glossaries, speaker dialogue dashes and natural spoken-tone punctuation. English translation is next.

## Principles
- **100% local.** No cloud APIs, accounts or telemetry. Audio, video and text never leave your machine. Works offline after a one-time model download.
- **Runs in Docker** with NVIDIA GPU support. No Python or CUDA setup on the host.
- **Built for review.** Machine output is a strong first draft. A human reviewer polishes it.

## Planned pipeline
```
video/audio → ffmpeg → VAD → Whisper large-v3 (tr) → forced alignment
  → (optional diarization) → segmentation → local LLM translation
  → subtitle formatting (readability rules) → SRT / VTT / ASS
```

## Target hardware
Designed for an NVIDIA GPU with 8 GB VRAM (e.g. RTX 3070), 14 GB RAM and 6 cores. Heavy models run one at a time. Bigger GPUs mean faster runs and larger translation models.

## Documentation
| Doc | Purpose |
|---|---|
| [product-requirements.md](docs/product-requirements.md) | Scope, requirements, quality targets, milestones |
| [architecture.md](docs/architecture.md) | Pipeline stages, stack, data model, API |
| [design-system.md](docs/design-system.md) | Review editor UI tokens and components |
| [agent-guide.md](docs/agent-guide.md) | Conventions for AI coding agents, Turkish-specific rules |
| [CLAUDE.md](CLAUDE.md) | Short project brief for Claude Code |

## Roadmap
1. **M1** Core CLI: ingest → ASR → Turkish SRT
2. **M2** Local LLM translation to English + subtitle formatter
3. **M3** Series glossary, character sheet, review flags
4. **M4** Web review editor
5. **M5** Diarization, music separation, more target languages, batch mode

## Legal
Use this tool only on media you have the right to process. It provides no content distribution features. Model licenses (Whisper, translation LLMs, pyannote) are governed by their own terms.

## Data sources and attribution
The series glossary (`glossary/series/`) combines metadata from TMDB, TheTVDB and TVmaze.
This product uses the TMDB API but is not endorsed or certified by TMDB. Metadata remains the property of
its providers; refresh it yourself with `subai glossary-fetch` using your own API keys. Reference subtitles
used for evaluation are not included in this repository.

## License
[MIT](LICENSE)
