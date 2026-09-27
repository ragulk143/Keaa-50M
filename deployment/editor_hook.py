"""
editor_hook.py — Minimal inference wrapper: loads Keaa-50M + its tokenizer
and generates an autocomplete continuation for a code prefix. This is the
piece an editor plugin would call; as shipped it's a CLI you can pipe a
prefix into to sanity-check the model after training.

Usage:
  python editor_hook.py --prefix "FROM python:3.11-slim\nRUN pip install "

Requires: torch, tokenizers, safetensors
Expects release/model.safetensors, release/config.json, and
tokenizer_out/tokenizer.json to exist (from train.py / custom_tokenizer.py).
"""

import argparse
import json
import pathlib

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from safetensors.torch import load_model

from architecture import MambaConfig, KeaaMamba

RELEASE_DIR = pathlib.Path("release")
TOKENIZER_PATH = pathlib.Path("tokenizer_out/tokenizer.json")


def load_model_and_tokenizer(device):
    with open(RELEASE_DIR / "config.json") as f:
        cfg_dict = json.load(f)
    cfg = MambaConfig(**cfg_dict)
    model = KeaaMamba(cfg)
    load_model(model, str(RELEASE_DIR / "model.safetensors"))
    model.to(device).eval()

    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
    return model, tokenizer


@torch.no_grad()
def generate(model, tokenizer, prefix: str, max_new_tokens=60, temperature=0.7, top_k=40, device="cpu"):
    ids = tokenizer.encode(prefix).ids
    x = torch.tensor([ids], dtype=torch.long, device=device)

    for _ in range(max_new_tokens):
        logits = model(x)["logits"][:, -1, :] / temperature
        if top_k:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = -float("inf")
        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        x = torch.cat([x, next_id], dim=1)

    return tokenizer.decode(x[0].tolist())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", type=str, required=True)
    parser.add_argument("--max_new_tokens", type=int, default=60)
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = load_model_and_tokenizer(device)

    output = generate(
        model, tokenizer, args.prefix,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        device=device,
    )
    print(output)


if __name__ == "__main__":
    main()
