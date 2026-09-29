"""
train.py — Trains Keaa-50M (architecture.py) on the tokenized corpus
(custom_tokenizer.py's output) from random initialization.

Works in two modes with the same file:
  single GPU / CPU :  python training/train.py
  multi GPU (DDP)  :  torchrun --nproc_per_node=2 training/train.py

Kaggle 2 x T4 launch (from a notebook cell):
  !PYTHONPATH=model NCCL_P2P_DISABLE=1 KEAA_BATCH_SIZE=4 KEAA_GRAD_ACCUM=8 \
   KEAA_MAX_STEPS=3000 torchrun --nproc_per_node=2 training/train.py

All hyperparameters below can be overridden with KEAA_* environment variables.
BATCH_SIZE and GRAD_ACCUM_STEPS are PER GPU, so
  tokens per optimizer step = BATCH_SIZE * GRAD_ACCUM_STEPS * WORLD_SIZE * SEQ_LEN.

Checkpoints every CHECKPOINT_EVERY steps (rank 0 only, atomic write, last 2 kept)
and resumes automatically -- Kaggle/Colab can kill your session mid-run.

Expects, relative to the working directory:
  raw_data/                      <- from pull_data.py
  tokenizer_out/tokenizer.json   <- from custom_tokenizer.py
"""

import contextlib
import datetime
import json
import math
import os
import pathlib
import random
import shutil
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from tokenizers import Tokenizer
from safetensors.torch import save_model

from architecture import MambaConfig, KeaaMamba, count_parameters


def _env(name, default, cast=int):
    v = os.environ.get(name)
    return cast(v) if v not in (None, "") else default


# ---- paths ----
RAW_DATA_DIR = pathlib.Path("raw_data")
TOKENIZER_PATH = pathlib.Path("tokenizer_out/tokenizer.json")
CHECKPOINT_DIR = pathlib.Path("checkpoints")
RELEASE_DIR = pathlib.Path("release")

# ---- training hyperparameters (per-GPU batch sizes) ----
SEQ_LEN = _env("KEAA_SEQ_LEN", 512)
BATCH_SIZE = _env("KEAA_BATCH_SIZE", 4)
GRAD_ACCUM_STEPS = _env("KEAA_GRAD_ACCUM", 8)
MAX_STEPS = _env("KEAA_MAX_STEPS", 3000)
LEARNING_RATE = _env("KEAA_LR", 3e-4, float)
WARMUP_STEPS = _env("KEAA_WARMUP_STEPS", min(500, max(10, MAX_STEPS // 20)))
CHECKPOINT_EVERY = _env("KEAA_CHECKPOINT_EVERY", max(100, MAX_STEPS // 10))
EVAL_EVERY = _env("KEAA_EVAL_EVERY", CHECKPOINT_EVERY)
LOG_EVERY = _env("KEAA_LOG_EVERY", 10)
VAL_FRACTION = _env("KEAA_VAL_FRACTION", 0.03, float)   # fraction of FILES held out
KEEP_CHECKPOINTS = 2

# optional overrides so you can smoke-test with a tiny model
D_MODEL = _env("KEAA_D_MODEL", 960)
N_LAYERS = _env("KEAA_N_LAYERS", 8)


class ChunkedTextDataset(Dataset):
    """Packs all files into one token stream (each file ends with <eos>),
    then slices it into windows of seq_len + 1 tokens (the extra token is the
    label for the last position). Stored as one int32 tensor."""

    def __init__(self, files, tokenizer: Tokenizer, seq_len: int):
        eos_id = tokenizer.token_to_id("<eos>")
        stream = []
        for f in files:
            text = f.read_text(encoding="utf-8", errors="ignore")
            stream.extend(tokenizer.encode(text).ids + [eos_id])
        self.n_tokens = len(stream)
        n = (len(stream) - 1) // seq_len
        if n < 1:
            self.data = torch.empty(0, seq_len + 1, dtype=torch.int32)
            return
        t = torch.tensor(stream, dtype=torch.int32)
        # window i covers stream[i*seq_len : i*seq_len + seq_len + 1]
        self.data = torch.stack([t[i * seq_len : i * seq_len + seq_len + 1] for i in range(n)])

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        ids = self.data[idx].long()
        return ids[:-1], ids[1:]          # (input, next-token target)


def split_files(data_dir: pathlib.Path):
    """Deterministic file-level train/val split (identical on every rank)."""
    files = sorted(data_dir.rglob("*.txt"))
    if not files:
        raise FileNotFoundError(f"No .txt files under {data_dir}/. Run pull_data.py first.")
    random.Random(1337).shuffle(files)
    n_val = int(len(files) * VAL_FRACTION)
    if len(files) >= 20:
        n_val = max(n_val, 2)
    return files[n_val:], files[:n_val]


def lm_loss(logits, targets, reduction="mean"):
    # fp32 cross-entropy. `targets` are ALREADY next-token targets (see dataset),
    # so there is NO extra shift here. (KeaaMamba's built-in `labels=` path shifts
    # a second time, which would train the model to predict token t+2.)
    return F.cross_entropy(
        logits.float().reshape(-1, logits.size(-1)), targets.reshape(-1), reduction=reduction
    )


def cosine_lr(step, warmup_steps, max_steps, base_lr, min_ratio=0.1):
    if step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    progress = min((step - warmup_steps) / max(1, max_steps - warmup_steps), 1.0)
    return base_lr * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress)))


def find_latest_checkpoint():
    if not CHECKPOINT_DIR.exists():
        return None
    ckpts = sorted(CHECKPOINT_DIR.glob("step_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    return ckpts[-1] if ckpts else None


def setup_distributed():
    ddp = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    if ddp:
        rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
            backend = "nccl"
        else:
            device, backend = torch.device("cpu"), "gloo"
        dist.init_process_group(backend, timeout=datetime.timedelta(minutes=30))
    else:
        rank, world, local_rank = 0, 1, 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return ddp, rank, world, local_rank, device


@torch.no_grad()
def evaluate(raw_model, val_data, device, rank, world, ddp, use_amp, batch_size=8):
    """Validation loss, sharded across ranks and all-reduced. Every rank must call this."""
    raw_model.eval()
    total = torch.zeros(2, dtype=torch.float64, device=device)   # [sum loss, n tokens]
    for i in range(rank * batch_size, len(val_data), world * batch_size):
        chunk = val_data[i : i + batch_size].long().to(device)
        x, y = chunk[:, :-1], chunk[:, 1:]
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits = raw_model(x)["logits"]
        total[0] += lm_loss(logits, y, reduction="sum").double()
        total[1] += y.numel()
    if ddp:
        dist.all_reduce(total)
    raw_model.train()
    return (total[0] / total[1].clamp(min=1)).item()


def save_checkpoint(raw_model, optimizer, scaler, step):
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    final = CHECKPOINT_DIR / f"step_{step}.pt"
    tmp = CHECKPOINT_DIR / f"step_{step}.pt.tmp"
    torch.save({"model": raw_model.state_dict(), "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(), "step": step}, tmp)
    os.replace(tmp, final)                      # atomic: never leaves a half-written ckpt
    old = sorted(CHECKPOINT_DIR.glob("step_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    for p in old[:-KEEP_CHECKPOINTS]:
        p.unlink()
    print(f"Saved checkpoint: {final}", flush=True)


def main():
    ddp, rank, world, local_rank, device = setup_distributed()
    is_main = rank == 0
    use_amp = device.type == "cuda"
    torch.manual_seed(1234)                     # same init on every rank (DDP also broadcasts)

    def log(*a):
        if is_main:
            print(*a, flush=True)

    if not use_amp:
        log("WARNING: no GPU detected. This will be extremely slow -- "
            "make sure the Kaggle/Colab GPU accelerator is actually enabled.")
    log(f"world_size={world} device={device} ddp={ddp}")

    if is_main:
        RELEASE_DIR.mkdir(exist_ok=True)
        CHECKPOINT_DIR.mkdir(exist_ok=True)

    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
    cfg = MambaConfig(vocab_size=tokenizer.get_vocab_size(), d_model=D_MODEL, n_layers=N_LAYERS)
    raw_model = KeaaMamba(cfg).to(device)
    log(f"Model: {count_parameters(raw_model):,} params "
        f"(d_model={cfg.d_model}, n_layers={cfg.n_layers}, vocab_size={cfg.vocab_size})")

    train_files, val_files = split_files(RAW_DATA_DIR)
    train_ds = ChunkedTextDataset(train_files, tokenizer, SEQ_LEN)
    val_ds = ChunkedTextDataset(val_files, tokenizer, SEQ_LEN) if val_files else None
    if len(train_ds) < BATCH_SIZE * world:
        raise ValueError(f"Only {len(train_ds)} training chunks, need >= BATCH_SIZE*world_size "
                         f"({BATCH_SIZE * world}). Lower KEAA_SEQ_LEN/KEAA_BATCH_SIZE or add data.")
    have_val = val_ds is not None and len(val_ds) > 0

    tokens_per_step = BATCH_SIZE * GRAD_ACCUM_STEPS * world * SEQ_LEN
    steps_per_epoch = len(train_ds) / (BATCH_SIZE * GRAD_ACCUM_STEPS * world)
    log(f"Train: {len(train_files)} files, {len(train_ds)} chunks, ~{train_ds.n_tokens/1e6:.2f}M tokens | "
        f"Val: {len(val_files)} files, {len(val_ds) if have_val else 0} chunks")
    log(f"{tokens_per_step:,} tokens/step -> {steps_per_epoch:.1f} steps/epoch -> "
        f"{MAX_STEPS / steps_per_epoch:.1f} epochs at MAX_STEPS={MAX_STEPS}")
    if MAX_STEPS / steps_per_epoch > 4:
        log("WARNING: >4 epochs over this corpus. A 50M model will memorize a small dataset; "
            "expect val loss to bottom out and rise. More data beats more steps.")

    # no weight decay on norms, biases, embeddings-tied head, A_log, D
    decay, no_decay = [], []
    for n, p in raw_model.named_parameters():
        (no_decay if (p.ndim < 2 or n.endswith("A_log") or n.endswith(".D") or "embedding" in n)
         else decay).append(p)
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": 0.1}, {"params": no_decay, "weight_decay": 0.0}],
        lr=LEARNING_RATE, betas=(0.9, 0.95))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    start_step = 0
    ckpt_path = find_latest_checkpoint()
    if ckpt_path is not None:
        log(f"Resuming from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        raw_model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if "scaler" in ckpt:
            scaler.load_state_dict(ckpt["scaler"])
        start_step = ckpt["step"]

    if _env("KEAA_COMPILE", 0):
        log("Compiling model with torch.compile (first step will be slow -- that's the compile "
            "warmup, not the real per-step time)")
        raw_model = torch.compile(raw_model)          # compile BEFORE DDP wrap, per torch docs

    model = DDP(raw_model, device_ids=[local_rank] if use_amp else None) if ddp else raw_model

    sampler = DistributedSampler(train_ds, num_replicas=world, rank=rank, shuffle=True,
                                 drop_last=True, seed=1234) if ddp else None
    loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,
                        shuffle=(sampler is None), drop_last=True, pin_memory=use_amp)
    epoch = start_step                          # different shuffle after each resume
    if sampler:
        sampler.set_epoch(epoch)
    data_iter = iter(loader)

    model.train()
    step = start_step
    bad_steps = 0
    t0 = time.time()

    while step < MAX_STEPS:
        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0

        for micro in range(GRAD_ACCUM_STEPS):
            try:
                x, y = next(data_iter)
            except StopIteration:
                epoch += 1
                if sampler:
                    sampler.set_epoch(epoch)
                data_iter = iter(loader)
                x, y = next(data_iter)
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

            # skip the gradient all-reduce on all but the last micro-batch
            sync = model.no_sync() if (ddp and micro < GRAD_ACCUM_STEPS - 1) else contextlib.nullcontext()
            with sync:
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                    logits = model(x)["logits"]
                loss = lm_loss(logits, y) / GRAD_ACCUM_STEPS
                scaler.scale(loss).backward()
            accum_loss += loss.item()

        scaler.unscale_(optimizer)
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        lr = cosine_lr(step, WARMUP_STEPS, MAX_STEPS, LEARNING_RATE)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        scaler.step(optimizer)
        scaler.update()
        step += 1

        if step % LOG_EVERY == 0 or step == 1:
            elapsed = time.time() - t0
            done = step - start_step
            eta = elapsed / done * (MAX_STEPS - step)
            log(f"step {step:6d}/{MAX_STEPS} | loss {accum_loss:.4f} | gnorm {float(gnorm):.2f} | "
                f"lr {lr:.2e} | {elapsed/done:.1f}s/step | eta {eta/3600:.2f}h")
        bad_steps = bad_steps + 1 if not math.isfinite(accum_loss) else 0
        if bad_steps >= 10:
            raise RuntimeError("Loss has been NaN/inf for 10 steps in a row -- "
                               "likely fp16 overflow in the scan; see notes.")

        if have_val and (step % EVAL_EVERY == 0 or step == MAX_STEPS):
            vl = evaluate(raw_model, val_ds.data, device, rank, world, ddp, use_amp)
            log(f"  >> step {step} val_loss {vl:.4f} | val_ppl {math.exp(min(vl, 20)):.2f}")

        if is_main and (step % CHECKPOINT_EVERY == 0):
            save_checkpoint(raw_model, optimizer, scaler, step)

    # ---- final export for release (rank 0 only) ----
    if is_main:
        save_model(raw_model, str(RELEASE_DIR / "model.safetensors"))
        with open(RELEASE_DIR / "config.json", "w") as f:
            json.dump({
                "vocab_size": cfg.vocab_size, "d_model": cfg.d_model, "n_layers": cfg.n_layers,
                "d_state": cfg.d_state, "d_conv": cfg.d_conv, "expand": cfg.expand,
                "dt_rank": cfg.dt_rank,
            }, f, indent=2)
        shutil.copy(TOKENIZER_PATH, RELEASE_DIR / "tokenizer.json")   # weights are useless without it
        print(f"Training complete. Exported to {RELEASE_DIR}/ "
              f"(model.safetensors + config.json + tokenizer.json).", flush=True)

    if ddp:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
