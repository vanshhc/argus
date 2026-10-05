"""Test inputs shared by the comparison scripts."""

from pathlib import Path

import torch
from transformers import AutoTokenizer

TEXT_FILES = ("LICENSE.txt", "USE_POLICY.md")


def text_tokens(hf_dir: Path, limit: int) -> torch.Tensor:
    """Real text that ships with the model, so tests need no other data."""
    tokenizer = AutoTokenizer.from_pretrained(hf_dir)
    text = "\n\n".join((hf_dir / name).read_text() for name in TEXT_FILES if (hf_dir / name).exists())
    if not text:
        raise FileNotFoundError(f"None of {TEXT_FILES} in {hf_dir}")
    return tokenizer(text, return_tensors="pt").input_ids[:, :limit]
