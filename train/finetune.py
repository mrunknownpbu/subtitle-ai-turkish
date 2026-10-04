#!/usr/bin/env python3
"""QLoRA fine-tune of Whisper large-v3 on the windows built by prepare.py (fits an 8 GB GPU).

  python /repo/train/finetune.py --data /repo/workspace/train --out /repo/workspace/train/lora --epochs 4

Then merge + convert for faster-whisper:
  python /repo/train/finetune.py --merge /repo/workspace/train/lora --out /repo/workspace/train/merged
  ct2-transformers-converter --model /repo/workspace/train/merged --output_dir /repo/workspace/train/ct2 \
      --copy_files tokenizer.json preprocessor_config.json --quantization float16
"""
import argparse
import json
import random
import time
import wave
from pathlib import Path

import numpy as np
import torch
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig, WhisperForConditionalGeneration, WhisperProcessor

BASE = "openai/whisper-large-v3"


def read_wav(p: str) -> np.ndarray:
    with wave.open(p, "rb") as w:
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0


def merge(adapter: Path, out: Path) -> None:
    model = WhisperForConditionalGeneration.from_pretrained(BASE, torch_dtype=torch.float16)
    model = PeftModel.from_pretrained(model, str(adapter)).merge_and_unload()
    model.save_pretrained(out, safe_serialization=True)
    WhisperProcessor.from_pretrained(BASE).save_pretrained(out)
    print("merged model saved to", out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--merge", type=Path, help="merge this adapter into the base model and save to --out")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--max-steps", type=int, default=0, help="stop early (smoke test)")
    a = ap.parse_args()
    if a.merge:
        return merge(a.merge, a.out)

    torch.manual_seed(0); random.seed(0)
    proc = WhisperProcessor.from_pretrained(BASE, language="turkish", task="transcribe")
    tok = proc.tokenizer
    tok.set_prefix_tokens(language="turkish", task="transcribe", predict_timestamps=False)
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                             bnb_4bit_compute_dtype=torch.float16)
    model = WhisperForConditionalGeneration.from_pretrained(BASE, quantization_config=bnb, device_map={"": 0})
    model.config.use_cache = False
    # non-reentrant checkpointing: with the default (reentrant) variant the ENCODER's LoRA layers get no
    # gradients (its inputs come from frozen conv layers), so only the decoder would train.
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True,
                                            gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.05, bias="none",
                                             target_modules=["q_proj", "k_proj", "v_proj", "out_proj"]))
    model.print_trainable_parameters()

    rows = [json.loads(l) for l in open(a.data / "train.jsonl", encoding="utf-8")]
    print(f"{len(rows)} training windows")
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.01)
    total = max(1, (len(rows) // (a.batch * a.accum)) * a.epochs)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 8) * max(0.05, 1 - s / total))
    scaler = torch.amp.GradScaler("cuda")
    start_id = model.config.decoder_start_token_id
    step, t0 = 0, time.time()
    model.train()
    for epoch in range(a.epochs):
        random.shuffle(rows)
        run_loss, n = 0.0, 0
        for i in range(0, len(rows) - a.batch + 1, a.batch):
            batch = rows[i:i + a.batch]
            feats = proc.feature_extractor([read_wav(r["audio"]) for r in batch], sampling_rate=16000,
                                           return_tensors="pt").input_features.to("cuda", torch.float16)
            ids = [tok(r["text"]).input_ids for r in batch]
            ids = [x[1:] if x and x[0] == start_id else x for x in ids]
            width = max(len(x) for x in ids)
            labels = torch.tensor([x + [-100] * (width - len(x)) for x in ids]).to("cuda")
            with torch.autocast("cuda", dtype=torch.float16):
                loss = model(input_features=feats, labels=labels).loss / a.accum
            scaler.scale(loss).backward()
            run_loss += loss.item() * a.accum; n += 1
            if n % a.accum == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); sched.step(); step += 1
                if step % 5 == 0 or step <= 2:
                    print(f"epoch {epoch + 1} step {step}/{total} loss {run_loss / n:.4f} "
                          f"lr {sched.get_last_lr()[0]:.2e} {time.time() - t0:.0f}s "
                          f"vram {torch.cuda.max_memory_allocated() / 1e9:.1f}GB", flush=True)
                if a.max_steps and step >= a.max_steps:
                    break
        print(f"== epoch {epoch + 1} mean loss {run_loss / max(n, 1):.4f}", flush=True)
        if a.max_steps and step >= a.max_steps:
            break
    model.save_pretrained(a.out)
    print("adapter saved to", a.out)


if __name__ == "__main__":
    main()
