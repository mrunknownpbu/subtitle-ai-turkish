# Translation (Turkish SRT → English SRT)

`./scripts/subai translate -i "<name>.tr.srt" --series tvdb-383383` writes `<name>.en.srt`, one English cue per Turkish cue
(same timings; merging and splitting belong to Format). It runs entirely on the GPU, one model at a time.

## Design
1. **Draft** (`--draft opus`, default): `Helsinki-NLP/opus-mt-tc-big-tr-en` translates every cue (0.4 GB, ~25 s per episode).
   Other values: `none` (LLM only) or an English `.srt` with the same cues.
2. **Edit**: Qwen3 8B (Ollama, Q4, 100% GPU) edits the draft in batches of 8 cues with 3 cues of context on each side,
   the series synopsis, characters with gender, fixed translations and never-translate terms from the glossary.
   JSON-schema output, temperature 0, 3 retries, then single-cue fallback, then an error.
3. **Post-fix** (`fix_honorifics`): Madam/Mr/Mrs/Sir become Hanım/Bey when the Turkish has them, and "Bey Evren" becomes "Evren Bey".

Ollama runs as the `ollama` compose service (volume `subai-ollama`, `127.0.0.1:11434`, one model loaded). Start it with
`docker compose -f docker/compose.yaml up -d ollama` and stop it before Whisper jobs: they do not share the GPU well.
Prompt rules are in `system_prompt()`; change `PROMPT_VERSION` when they change.

## Measurements (S01E05, 1,384 cues paired with the human English subtitle, corpus chrF)
Score with `python3 eval/chrf.py --ref "<ep>.en.hi.srt" --hyp A.en.srt --hyp B.en.srt` (paired bootstrap against the first).
The human English subtitle is for scoring only, never pipeline input.

| System | chrF | Time |
|---|---|---|
| Qwen3 8B alone | 48.5 | ~23 min |
| opus-mt alone | 51.4 | 24 s |
| opus-mt draft edited by Qwen3, no post-fix | 51.7 | ~20 min |
| **Shipped: draft + edit + honorific post-fix** | **52.2** (+0.8 vs opus-mt alone, interval +0.2 to +1.5) | ~14 min |

chrF rewards literal overlap with a paraphrasing human subtitle, so it understates the edit pass, which fixes names
("Mr. Universe" → Evren Bey), honorifics and dropped content. Gemma 3 12B read best but scored lowest, and runs 15-35% on the CPU
at 8 GB, which is why it is not used. Gemma 3 4B fails the JSON schema. NLLB-200 1.3B/3.3B hallucinate on fragments.
subtitle-ai (the sibling project) measured opus-mt +6 to +7 chrF over NLLB on 16 episodes; this is the same model.

## Known limits
- One episode, one series. Idioms and some questions are still literal ("Tamam mısın?" → "Are you okay?").
- Output varies slightly between runs even at temperature 0 (batch context).
- The edit pass rewrites about 44% of cues; most are small wording changes.
