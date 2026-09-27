"""
train.py — Trains Keaa-50M (architecture.py) on the tokenized corpus
(custom_tokenizer.py's output) for real, from random initialization.

Run this on Kaggle (Settings -> Accelerator -> GPU T4 x2) or Colab
(Runtime -> Change runtime type -> GPU).

Checkpoints every CHECKPOINT_EVERY steps to checkpoints/, and resumes
automatically if a checkpoint exists -- both Kaggle and Colab can cut your
session off mid-run, so this is not optional.

Expects, relative to the working directory:
  raw_data/          <- from pull_data.py
  tokenizer_out/tokenizer.json  <- from custom_tokenizer.py

Requires: torch, tokenizers, safetensors
"""

import json
import math
import pathlib
import time

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer
from safetensors.torch import save_model

from architecture import MambaConfig, KeaaMamba, count_parameters

# ---- paths ----
RAW_DATA_DIR = pathlib.Path("raw_data")
TOKENIZER_PATH = pathlib.Path("tokenizer_out/tokenizer.json")
CHECKPOINT_DIR = pathlib.Path("checkpoints")
RELEASE_DIR = pathlib.Path("release")

# ---- training hyperparameters ----
SEQ_LEN = 512
BATCH_SIZE = 16
GRAD_ACCUM_STEPS = 4          # effective batch size = BATCH_SIZE * GRAD_ACCUM_STEPS = 64
MAX_STEPS = 20000             # adjust down for a shorter first run; this is the ceiling, not a target
LEARNING_RATE = 3e-4
WARMUP_STEPS = 500
CHECKPOINT_EVERY = 500
LOG_EVERY = 20


class ChunkedTextDataset(Dataset):
    """Tokenizes every file once, then slices it into fixed-length windows.
    Simple and memory-cheap enough for a corpus of a few thousand short
    scripts; if your corpus grows past a few hundred MB, switch this to
    streaming instead of loading every chunk into a Python list up front.
    """

    def __init__(self, data_dir: pathlib.Path, tokenizer: Tokenizer, seq_len: int):
        self.seq_len = seq_len
        self.chunks = []

        files = sorted(data_dir.rglob("*.txt"))
        if not files:
            raise FileNotFoundError(f"No .txt files under {data_dir}/. Run pull_data.py first.")

        for f in files:
            text = f.read_text(encoding="utf-8", errors="ignore")
            ids = tokenizer.encode(text).ids
            # need seq_len + 1 tokens per chunk (last token is only a label, never an input)
            for i in range(0, len(ids) - seq_len, seq_len):
                self.chunks.append(ids[i : i + seq_len + 1])

        if not self.chunks:
            raise ValueError(
                "No training chunks produced -- your files are shorter than SEQ_LEN. "
                "Lower SEQ_LEN or collect longer files."
            )

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        ids = self.chunks[idx]
        x = torch.tensor(ids[:-1], dtype=torch.long)
        y = torch.tensor(ids[1:], dtype=torch.long)
        return x, y


def cosine_lr(step, warmup_steps, max_steps, base_lr):
    if step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    return 0.5 * base_lr * (1 + math.cos(math.pi * min(progress, 1.0)))


def find_latest_checkpoint():
    if not CHECKPOINT_DIR.exists():
        return None
    ckpts = sorted(CHECKPOINT_DIR.glob("step_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    return ckpts[-1] if ckpts else None


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: no GPU detected. This will be extremely slow -- "
              "make sure the Kaggle/Colab GPU accelerator is actually enabled.")

    CHECKPOINT_DIR.mkdir(exist_ok=True)
    RELEASE_DIR.mkdir(exist_ok=True)

    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
    cfg = MambaConfig(vocab_size=tokenizer.get_vocab_size())
    model = KeaaMamba(cfg).to(device)
    print(f"Model: {count_parameters(model):,} params "
          f"(d_model={cfg.d_model}, n_layers={cfg.n_layers}, vocab_size={cfg.vocab_size})")

    dataset = ChunkedTextDataset(RAW_DATA_DIR, tokenizer, SEQ_LEN)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    print(f"Dataset: {len(dataset)} chunks of {SEQ_LEN} tokens "
          f"(~{len(dataset) * SEQ_LEN / 1e6:.1f}M tokens)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

    start_step = 0
    ckpt_path = find_latest_checkpoint()
    if ckpt_path is not None:
        print(f"Resuming from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step = ckpt["step"]

    model.train()
    step = start_step
    data_iter = iter(loader)
    t0 = time.time()

    while step < MAX_STEPS:
        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0

        for _ in range(GRAD_ACCUM_STEPS):
            try:
                x, y = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                x, y = next(data_iter)
            x, y = x.to(device), y.to(device)

            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(device == "cuda")):
                out = model(x, labels=y)
                loss = out["loss"] / GRAD_ACCUM_STEPS

            scaler.scale(loss).backward()
            accum_loss += loss.item()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        lr = cosine_lr(step, WARMUP_STEPS, MAX_STEPS, LEARNING_RATE)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        scaler.step(optimizer)
        scaler.update()

        if step % LOG_EVERY == 0:
            elapsed = time.time() - t0
            print(f"step {step:6d} | loss {accum_loss:.4f} | lr {lr:.2e} | {elapsed:.0f}s elapsed")

        if step > 0 and step % CHECKPOINT_EVERY == 0:
            ckpt_file = CHECKPOINT_DIR / f"step_{step}.pt"
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step,
            }, ckpt_file)
            print(f"Saved checkpoint: {ckpt_file}")

        step += 1

    # ---- final export for release ----
    save_model(model, str(RELEASE_DIR / "model.safetensors"))
    with open(RELEASE_DIR / "config.json", "w") as f:
        json.dump({
            "vocab_size": cfg.vocab_size,
            "d_model": cfg.d_model,
            "n_layers": cfg.n_layers,
            "d_state": cfg.d_state,
            "d_conv": cfg.d_conv,
            "expand": cfg.expand,
            "dt_rank": cfg.dt_rank,
        }, f, indent=2)
    print(f"Training complete. Exported to {RELEASE_DIR}/ "
          f"(model.safetensors + config.json) -- ready to push to Hugging Face.")


if __name__ == "__main__":
    main()
