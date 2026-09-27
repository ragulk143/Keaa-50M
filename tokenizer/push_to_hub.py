"""
push_to_hub.py — Uploads release/model.safetensors + release/config.json
(and a model card) to a Hugging Face model repo.

Run this after train.py finishes, from Kaggle/Colab (fastest, no download/
re-upload round trip) or locally once you've pulled release/ down.

Requires: huggingface_hub
  pip install huggingface_hub
Log in first (paste a token with write access, from huggingface.co/settings/tokens):
  huggingface-cli login
"""

import pathlib
from huggingface_hub import HfApi, create_repo

REPO_ID = "your-username/keaa-50m"  # <-- change this
RELEASE_DIR = pathlib.Path("release")

MODEL_CARD = """---
license: apache-2.0
tags:
  - mamba
  - state-space-model
  - code-generation
  - autocomplete
  - infrastructure-as-code
---

# Keaa-50M

A 50M-parameter Mamba (S6) state-space model trained from scratch for local,
zero-cost autocomplete on infrastructure automation code: GitHub Actions
YAML, Dockerfiles, and Bash scripts.

- Architecture: custom Mamba/S6 (d_model=960, n_layers=8, d_state=16), no
  attention -- linear O(n) memory scaling instead of a Transformer's O(n^2).
- Trained on public GitHub workflow/Dockerfile/Bash files, from random
  initialization, on free Kaggle/Colab T4 GPUs.
- Runs on CPU for inference; the full weights file is well under 200MB.

Code, training scripts, and data pipeline: https://github.com/your-username/keaa-50m

## Usage
See `deployment/editor_hook.py` in the GitHub repo for a minimal
generation example using this checkpoint + the matching tokenizer.

## Status
Early release -- see the GitHub repo's README for current training step
count and known limitations before relying on this for anything production.
"""


def main():
    if not (RELEASE_DIR / "model.safetensors").exists():
        raise FileNotFoundError(
            f"{RELEASE_DIR}/model.safetensors not found. Run train.py first."
        )

    api = HfApi()
    create_repo(REPO_ID, repo_type="model", exist_ok=True)

    card_path = RELEASE_DIR / "README.md"
    card_path.write_text(MODEL_CARD, encoding="utf-8")

    api.upload_folder(
        folder_path=str(RELEASE_DIR),
        repo_id=REPO_ID,
        repo_type="model",
        commit_message="Upload Keaa-50M weights + config",
    )
    print(f"Pushed to https://huggingface.co/{REPO_ID}")


if __name__ == "__main__":
    main()
