# Fine-tuning Whisper on the series (QLoRA)

Result: a series-specific Whisper large-v3 LoRA that lowers content WER by about 1 point on unseen
episodes and writes punctuation the way the human subtitlers do. Everything runs locally on the 8 GB GPU.

## Why labels are built the way they are
The human Turkish subtitles are condensed and written-standard (no fillers, "vallahi" for spoken "valla"),
so training on them directly would train the natural spoken tone away. `train/prepare.py` therefore builds
labels from **our own transcript, corrected by the reference**:

| alignment result | label gets |
|---|---|
| same word (spelling variants count as equal) | **our word and spelling**, the **reference's casing and punctuation** |
| different word | the reference word, unless either side is a discourse filler |
| extra word in ours | kept (assumed real speech: "ya", "yani") |
| extra word in the reference | added only if it is a short backchannel Whisper drops ("evet", "tamam" ...) |

Windows are skipped when alignment is poor (content WER > 0.4) or they contain song lyrics / quoted text.
Cue-break "..." and dialogue dashes are stripped from the reference (subtitle conventions, not speech).
Labels end up with the reference's punctuation rates (7.2 commas, 1.0 exclamations, 5.7 question marks per 100 words).

## Workflow
```
docker compose -f docker/compose.yml build subai
docker build -f docker/train.Dockerfile -t subai-train .                       # QLoRA tooling on top of subai
docker run --rm -u $(id -u):$(id -g) -v $PWD:/repo -v /data:/data:ro --entrypoint python subai:latest \
    /repo/train/prepare.py --episodes 1 2 3 --out /repo/workspace/train_v2      # 605 windows, 4.1 h
docker run --rm --gpus all -u $(id -u):$(id -g) -e HF_HOME=/models -v docker_subai-models:/models -v $PWD:/repo \
    --entrypoint python subai-train /repo/train/finetune.py --data /repo/workspace/train_v2 \
    --out /repo/workspace/train_v2/lora --epochs 3 --lr 3e-4                    # ~18 min, 2.5 GB VRAM
# merge + convert to CTranslate2 (needs transformers >= 4.56 for the converter):
docker run ... subai-train /repo/train/finetune.py --merge /repo/workspace/train_v2/lora --out /repo/workspace/train_v2/merged
cp <base snapshot>/tokenizer.json <base snapshot>/preprocessor_config.json workspace/train_v2/merged/
pip install "transformers>=4.56,<5" && ct2-transformers-converter --model workspace/train_v2/merged \
    --output_dir workspace/models/<name> --copy_files tokenizer.json preprocessor_config.json --quantization float16
```
Use it: `./scripts/subai run ... --model /ft/<name>` (`workspace/models` is mounted read-only at `/ft`).

## Results (held out: 4 x 10 min from S01E04 and S01E05, never trained on)
| | content WER | strict WER | commas | ! | ? (per 100 words; reference 7.52 / 1.04 / 6.35) |
|---|---|---|---|---|---|
| large-v3 baseline | 14.47% | 16.07% | 3.23 | 0.03 | 4.20 |
| A: labels with our punctuation | 13.64% | 15.48% | 1.73 | 0.00 | 4.46 |
| **B: labels with reference punctuation** (`sen-cal-kapimi-lora-v1`) | **13.49%** | **15.36%** | **7.93** | **0.37** | **5.26** |

Per chunk (baseline -> B): 17.0 -> 13.5, 16.6 -> 16.8, 10.7 -> 11.3, 12.5 -> 11.2. Cue-boundary F1 vs human breaks
and timing offsets are unchanged (0.77 / 0.75 vs 0.69 / 0.77), no new hallucination phrases.

## Caveats
* Evidence is 40 minutes from two episodes of one series; one chunk got slightly worse. Gains on other series are untested and may be smaller.
* `sen-cal-kapimi-lora-v2` is trained on S01E01-05 with the identical recipe and **has no held-out validation**.
* Pitfalls found: reentrant gradient checkpointing leaves the encoder's LoRA layers untrained (use `use_reentrant=False`);
  the CTranslate2 converter needs a newer transformers than the training stack; merged models lack `tokenizer.json`.
* Decoding prompts, fp16, wider beam, and vocal separation (Demucs) were tested first and did not help (see architecture.md decisions 13 and below).
