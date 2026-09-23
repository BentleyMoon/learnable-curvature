"""Download the corpora into data/. They are not committed.

WikiText-2 (raw, v1) is the only corpus any claim in CLAIMS.md depends on. The file is built the way
the original train_bpe.py built it: train, validation and test splits in that order, non-empty lines
only, one newline after each. Its SHA-256 is checked against the file the runs used.

The March language runs (pilot, Phase A, Phase B, ablations, sweep) loaded WikiText-2 straight from
Hugging Face inside learnable_curvature_transformer.py, so they need no file. The ternary-lane runs
(T0-T3) and this repo's reruns read data/wikitext2_all.txt.

    python scripts/get_data.py            # WikiText-2 only
    python scripts/get_data.py --python   # also the Python code corpus (no claim depends on it)
"""
import argparse
import hashlib
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(REPO, "data")
WIKITEXT_SHA256 = "1022cd9a05c343e874dabddeb12eae68b27d5ceb904fb7915805ce04aeb22355"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def get_wikitext(out_dir):
    from datasets import load_dataset
    path = os.path.join(out_dir, "wikitext2_all.txt")
    ds = load_dataset("wikitext", "wikitext-2-raw-v1")
    with open(path, "w", encoding="utf-8") as f:
        for split in ("train", "validation", "test"):
            for item in ds[split]:
                text = item["text"]
                if text and text.strip():
                    f.write(text + "\n")
    digest = sha256(path)
    ok = digest == WIKITEXT_SHA256
    print("%s  %s  %s" % (path, digest, "matches the file the runs used" if ok else "DOES NOT MATCH"))
    return ok


def get_python_corpus(out_dir, target_chars=10_000_000):
    """The March code-corpus runs (code_char_4h2l). Streaming order is not guaranteed stable, so no hash."""
    from datasets import load_dataset
    ds = load_dataset("codeparrot/codeparrot-clean", split="train", streaming=True)
    chunks, total = [], 0
    for item in ds:
        text = item.get("content", "")
        if text:
            chunks.append(text)
            total += len(text)
            if total >= target_chars:
                break
    text = "\n".join(chunks)
    split = int(len(text) * 0.9)
    for name, part in (("python_code_train.txt", text[:split]), ("python_code_val.txt", text[split:])):
        with open(os.path.join(out_dir, name), "w", encoding="utf-8", errors="ignore") as f:
            f.write(part)
        print(os.path.join(out_dir, name), len(part), "chars")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--python", action="store_true", help="also download the Python code corpus")
    ap.add_argument("--out", default=DATA)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    ok = get_wikitext(args.out)
    if args.python:
        get_python_corpus(args.out)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
