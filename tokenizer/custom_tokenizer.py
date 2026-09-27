"""
custom_tokenizer.py — Trains a small byte-level BPE tokenizer over the
collected YAML / Dockerfile / Bash corpus for Keaa-50M.

Why byte-level BPE instead of a fixed keyword list (image:, uses:, run:...):
a fixed list chokes the moment it sees a token it's never seen -- a repo
name, a version pin, a hash, a variable name -- which is most of the
entropy in real scripts. Byte-level BPE learns the common structural
tokens AND falls back to raw bytes for anything novel, so it never breaks.
The vocab stays small anyway because the domain is narrow.

Requires: tokenizers   (pip install tokenizers)
"""

import pathlib
from tokenizers import ByteLevelBPETokenizer

RAW_DATA_DIR = pathlib.Path("raw_data")
OUT_DIR = pathlib.Path("tokenizer_out")
VOCAB_SIZE = 4096  # small on purpose -- narrow domain, not English prose

SPECIAL_TOKENS = [
    "<pad>", "<unk>", "<bos>", "<eos>",
    "<mask>",  # for the masked-LM objective train.py will use
]


def collect_corpus_files():
    files = sorted(str(p) for p in RAW_DATA_DIR.rglob("*.txt"))
    if not files:
        raise FileNotFoundError(
            f"No .txt files found under {RAW_DATA_DIR}/. Run pull_data.py first."
        )
    return files


def main():
    OUT_DIR.mkdir(exist_ok=True)
    files = collect_corpus_files()
    print(f"Training tokenizer on {len(files)} files...")

    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(
        files=files,
        vocab_size=VOCAB_SIZE,
        min_frequency=2,
        special_tokens=SPECIAL_TOKENS,
    )

    tokenizer.save_model(str(OUT_DIR))
    tokenizer.save(str(OUT_DIR / "tokenizer.json"))
    print(f"Saved tokenizer to {OUT_DIR}/ (vocab.json, merges.txt, tokenizer.json)")

    # quick sanity check
    sample = 'FROM python:3.11-slim\nRUN pip install -r requirements.txt\nCMD ["python", "app.py"]'
    encoded = tokenizer.encode(sample)
    print("\nSample encode:")
    print(sample)
    print("->", encoded.tokens)


if __name__ == "__main__":
    main()
