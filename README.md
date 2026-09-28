  # Keaa-50M
<img width="1672" height="941" alt="image" src="https://github.com/user-attachments/assets/5b2e4536-3687-4f8f-abca-04b2f9fe2ef0" />

A 50M-parameter Mamba (S6) state-space model, trained from scratch, for
local zero-cost autocomplete on infrastructure automation code: GitHub
Actions YAML, Dockerfiles, and Bash scripts.

No attention mechanism, so working memory scales linearly (O(n)) instead
of quadratically -- it can parse long scripts on CPU without the memory
wall a small Transformer would hit.

- Architecture: `model/architecture.py` — custom Mamba/S6 block in plain
  PyTorch (d_model=960, n_layers=8, d_state=16, ~50.9M params)
- Data: `data/pull_data.py` — pulls public workflow/Dockerfile/Bash files
  via the GitHub search API
- Tokenizer: `tokenizer/custom_tokenizer.py` — byte-level BPE, 4096 vocab
- Training: `training/train.py` — full training loop with checkpointing
  and resume, built for Kaggle/Colab session limits
- Inference: `deployment/editor_hook.py` — loads the trained weights and
  generates a continuation for a given code prefix

## Status
Early release. See commit history / training logs for current step count.
This is not yet claiming state-of-the-art results -- it's a from-scratch
SSM trained on a narrow domain with free-tier compute; treat early
checkpoints as a work in progress.

## Reproducing training
See "Training on Kaggle/Colab" below. Config lives in
`training/configs/keaa-50m.json`.

## Citation
See `CITATION.cff`. Cite the archived GitHub release (Zenodo DOI) for the
code and architecture; cite the Hugging Face repo for the specific weights.

## License
Apache-2.0
