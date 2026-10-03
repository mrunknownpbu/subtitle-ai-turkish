# Subtitle AI — Turkish Series

Fully local, AI-assisted subtitle generator for Turkish TV series. It transcribes Turkish speech, translates it (English first; Malay and Indonesian planned), and exports timed subtitles as SRT, WebVTT and ASS.

> **Status: planning.** This repo currently holds design docs only. No code yet.

## Principles
- **100% local.** No cloud APIs, accounts or telemetry. Audio, video and text never leave your machine. Works offline after a one-time model download.
- **No Docker.** Native install with `uv`, system `ffmpeg` and a native Ollama service.
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
| [product-requirements.md](product-requirements.md) | Scope, requirements, quality targets, milestones |
| [architecture.md](architecture.md) | Pipeline stages, stack, data model, API |
| [design-system.md](design-system.md) | Review editor UI tokens and components |
| [agent-guide.md](agent-guide.md) | Conventions for AI coding agents, Turkish-specific rules |
| [CLAUDE.md](CLAUDE.md) | Short project brief for Claude Code |

## Roadmap
1. **M1** Core CLI: ingest → ASR → Turkish SRT
2. **M2** Local LLM translation to English + subtitle formatter
3. **M3** Series glossary, character sheet, review flags
4. **M4** Web review editor
5. **M5** Diarization, music separation, more target languages, batch mode

## Legal
Use this tool only on media you have the right to process. It provides no content distribution features. Model licenses (Whisper, translation LLMs, pyannote) are governed by their own terms.

## License
[MIT](LICENSE)
