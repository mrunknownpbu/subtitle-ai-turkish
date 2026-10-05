# Translation (Turkish SRT → English SRT)

`./scripts/subai translate -i "<name>.tr.srt" --series tvdb-383383` writes `<name>.en.srt`, one English cue per Turkish cue
(same timings; merging and splitting belong to Format). It runs on the GPU in its own process: no LLM, no Ollama.

## Design
Model: `Helsinki-NLP/opus-mt-tc-big-tr-en` (0.4 GB, fp16, beam 2, no_repeat_ngram_size 4, batches of 32 longest-first;
a CUDA out-of-memory batch is halved). Loaded once per run and freed on exit. There is no CPU fallback.
Approach and thresholds follow the sibling project subtitle-ai. Per cue (`translate.py`, text steps in `protect.py`):

1. `collapse_repeats`: Whisper's "gel, gel, gel…" loops are cut to two.
2. **Split**: a cue whose every line starts with "-" is translated turn by turn; multi-sentence text is translated sentence by sentence
   (the model drops sentences and garbles two speakers when given them in one call). Results are rejoined into the same cue.
3. **Phrase map** (`phrase_map:` in `glossary/language/tr.yaml`): a sentence that is exactly "Peki", "Tamam", … gets the fixed
   English and never reaches the model (it invents continuations for one-word cues). Trailing `.!?` is ignored; İ/ı aware.
4. **Protect**: character names, first names and aliases, plus glossary terms, become placeholders (Xaa, Xab…). Policy `keep` terms and
   names restore as written; policy `translate` terms restore as their English (burs → scholarship). Longest form first.
   A sentence that is only a name ("Cenk.") skips the model.
5. **Translate**, then a **run-on retry**: 8+ words with no punctuation are also translated in 6-word chunks and the better
   result (fewer lost names, else not near-empty) is kept.
6. **Repair and restore**: a near-miss placeholder ("Xax" for "Xac") is fixed when unambiguous, placeholders are restored, and a name
   the model dropped is put back (`recover_dropped_entities` never calls the model).
7. `fix_honorifics`: Madam/Mr/Mrs/Sir become Hanım/Bey when the Turkish has them; "Bey Evren" becomes "Evren Bey". Then `wrap` at 42 characters.
8. **QC (advisory)**: empty output, leaked placeholder, unbalanced bracket, or a length ratio under 0.25 or over 3.5 is logged as a warning. Output is never changed.

## Measurements (corpus chrF against the human English subtitle, same Turkish input for both)
Score with `python3 eval/chrf.py --ref "<ep>.en.hi.srt" --hyp A.en.srt --hyp B.en.srt` (paired bootstrap against the first).
The human English subtitle is for scoring only, never pipeline input.

| Episode | Previous: opus-mt draft + Qwen3 8B edit + honorific post-fix | **Now** |
|---|---|---|
| S01E05 (1,384 paired cues) | 52.19 | **52.53** (+0.33, interval −0.45 to +0.97) |
| S02E01 (1,633 paired cues) | 52.55 | **53.13** (+0.58, interval −0.09 to +1.16) |

Earlier S01E05 numbers: Qwen3 8B alone 48.5, opus-mt alone 51.4 (our first plain draft), draft + Qwen edit 51.7. The Qwen edit was ~14 minutes
per episode and needed the Ollama service; the current pipeline matches or beats it. chrF rewards literal overlap with a paraphrasing
human subtitle, so it understates how well either pipeline reads. Spot check: the old output wrote "Sevil Hanım Hanım" in one S01E05 cue; the new one does not.
The opus-mt-over-NLLB gain (+6 to +7 chrF on 16 episodes) was measured in subtitle-ai.

## Known limits
- Idioms and some questions are still literal ("Tamam mısın?" → "Are you okay?"); the phrase map only covers whole short utterances.
- Names that are also ordinary words (Deniz, Melek, Kiraz) are protected case-insensitively, so a lowercase "deniz" (sea) can come out as the name.
  Add a case-sensitive option to `Entity` if the eval shows it.
- Turkish case suffixes without an apostrophe ("Serkanlar'a" is fine; "Serkanın") are not matched.
- Two episodes of one series were measured.
