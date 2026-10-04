"""A small random-weight Llama 3 model for tests that need no downloads.

Usage: ./run.sh python -m coreai_ports.tiny <output_dir>
"""

import sys
from pathlib import Path

import torch
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

LLAMA3_ROPE = {
    "rope_type": "llama3",
    "rope_theta": 500000.0,
    "factor": 32.0,
    "low_freq_factor": 1.0,
    "high_freq_factor": 4.0,
    "original_max_position_embeddings": 8192,
}


def tiny_config(*, tie_word_embeddings: bool = True, rope: dict | None = None, **overrides) -> LlamaConfig:
    """Small Llama config. head_dim 16 with theta 500000 covers all three Llama 3 RoPE bands."""
    values = dict(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=1024,
        rms_norm_eps=1e-5,
        tie_word_embeddings=tie_word_embeddings,
        rope_parameters=dict(rope or LLAMA3_ROPE),
        initializer_range=0.2,
    )
    values.update(overrides)
    return LlamaConfig(**values)


def tiny_tokenizer(vocab_size: int) -> PreTrainedTokenizerFast:
    """Word-level tokenizer with one word per token ID. It exists only so export can bundle it."""
    from tokenizers import Tokenizer, models, pre_tokenizers

    vocab = {f"t{i}": i for i in range(vocab_size)}
    backend = Tokenizer(models.WordLevel(vocab=vocab, unk_token="t0"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    return PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="t0", eos_token="t1")


def save_tiny_model(output_dir: Path, max_position_embeddings: int = 512, seed: int = 0) -> Path:
    config = tiny_config(max_position_embeddings=max_position_embeddings)
    torch.manual_seed(seed)
    LlamaForCausalLM(config).save_pretrained(output_dir)
    tiny_tokenizer(config.vocab_size).save_pretrained(output_dir)
    return output_dir


if __name__ == "__main__":
    print(save_tiny_model(Path(sys.argv[1])))
