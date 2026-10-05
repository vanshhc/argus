"""Llama for iOS, built on Apple's iOS Mistral port.

Llama and Mistral share layer names and block structure. The port changes one
part: RoPE. Llama 3.x rescales the RoPE frequencies (`rope_type: llama3`).
Apple's RoPE cache supports only the default frequencies.
"""

import torch
from transformers import LlamaConfig
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from transformers.models.llama.modeling_llama import LlamaForCausalLM as HFLlamaForCausalLM

from coreai_models.models.ios.mistral import MistralForCausalLMForiOS
from coreai_models.primitives.ios.rope import RoPECache

SUPPORTED_ROPE_TYPES = ("default", "llama3")


def rope_parameters(config: LlamaConfig) -> dict:
    """Return the RoPE parameters across transformers config layouts."""
    params = getattr(config, "rope_parameters", None) or getattr(config, "rope_scaling", None)
    return dict(params or {})


def check_config(config: LlamaConfig) -> None:
    """Stop with a clear error for config values this port does not implement."""
    problems = []
    rope_type = rope_parameters(config).get("rope_type", "default")
    if rope_type not in SUPPORTED_ROPE_TYPES:
        problems.append(f"rope_type {rope_type!r} (supported: {', '.join(SUPPORTED_ROPE_TYPES)})")
    if getattr(config, "attention_bias", False):
        problems.append("attention_bias=True")
    if getattr(config, "mlp_bias", False):
        problems.append("mlp_bias=True")
    if getattr(config, "hidden_act", "silu") != "silu":
        problems.append(f"hidden_act {config.hidden_act!r} (supported: 'silu')")
    if getattr(config, "pretraining_tp", 1) != 1:
        problems.append(f"pretraining_tp={config.pretraining_tp}")
    if problems:
        raise ValueError("Unsupported Llama config for the iOS port: " + "; ".join(problems))


class LlamaRoPECache(RoPECache):
    """Apple's iOS RoPE cache with frequencies from Hugging Face's RoPE functions."""

    def __init__(self, config: LlamaConfig, head_dim: int, max_cache_size: int) -> None:
        # The parent constructor computes the cache, so set the config first.
        self._config = config
        self._rope_type = rope_parameters(config).get("rope_type", "default")
        base = rope_parameters(config).get("rope_theta", getattr(config, "rope_theta", 500_000))
        super().__init__(head_dim, max_cache_size, base)

    def _inverse_frequencies(self) -> torch.Tensor:
        if self._rope_type == "default":
            exponents = torch.arange(0, self._head_dim, 2, dtype=torch.float32) / self._head_dim
            return 1.0 / (self._base**exponents)
        inv_freq, attention_scaling = ROPE_INIT_FUNCTIONS[self._rope_type](self._config, "cpu")
        if attention_scaling != 1.0:
            raise ValueError(f"rope_type {self._rope_type!r} needs attention scaling; not supported")
        return inv_freq.float()

    def _compute_sin_and_cos(self, dtype: torch.dtype = torch.float32) -> None:
        # Adapted from RoPECache._compute_sin_and_cos in Apple's coreai-models
        # (Copyright 2026 Apple Inc., BSD-3-Clause). Only the frequencies differ.
        with torch.device("cpu"):
            theta = self._inverse_frequencies()
            if self._use_hf_impl:
                theta = theta.to(dtype).float()
            seq_idx = torch.arange(end=self._max_cache_size, dtype=torch.int32)
            freqs = seq_idx[:, None] * theta
            emb = torch.concatenate((freqs, freqs), dim=-1)
            self.cos_cached = torch.nn.Buffer(torch.cos(emb).to(dtype=dtype), persistent=False)
            self.sin_cached = torch.nn.Buffer(torch.sin(emb).to(dtype=dtype), persistent=False)


class LlamaForCausalLMForiOS(MistralForCausalLMForiOS):
    _HF_MODEL_CLASS = HFLlamaForCausalLM

    def _init_model(self, config: LlamaConfig) -> None:
        check_config(config)
        super()._init_model(config)
        head_dim = (
            getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        )
        if not isinstance(getattr(self.extend, "rope", None), RoPECache):
            raise RuntimeError("Apple's MistralExtend no longer has a `rope` RoPECache attribute")
        self.extend.rope = LlamaRoPECache(config, head_dim, config.max_position_embeddings)
