"""Accuracy tests for the Llama iOS port with small random-weight models.

These tests need no downloaded weights. They compare the port with Hugging Face's
LlamaForCausalLM on the same random weights. Pass limits are in DECISIONS.md.
"""

import math
import os
from collections.abc import Iterator

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

from coreai_models.models.ios.mistral import MistralForCausalLMForiOS
from coreai_models.models.registry import get_model_entry
from coreai_models.primitives.ios.cache import KVCacheHandler
from argus_kit.llama_ios import LlamaForCausalLMForiOS, check_config
from argus_kit.register import register
from argus_kit.tiny import LLAMA3_ROPE
from argus_kit.tiny import tiny_config as base_tiny_config
from tests._runner_infra.testing_utils import (
    _construct_causal_mask,
    load_state_dict_from_ref_model,
    run_torch_prompt_extend_test_ios,
)

MAX_ABS_DIFF = 1e-3
CONTROL_MIN_DIFF = 1e-2
SEQ_LEN = 1000

@pytest.fixture(autouse=True, scope="module")
def use_hf_impl() -> Iterator[None]:
    """Use Apple's Hugging Face parity path, as Apple's own conversion tests do."""
    original = os.environ.get("USE_HF_IMPL")
    os.environ["USE_HF_IMPL"] = "true"
    yield
    if original is None:
        os.environ.pop("USE_HF_IMPL", None)
    else:
        os.environ["USE_HF_IMPL"] = original


def tiny_config(**kwargs) -> LlamaConfig:
    return base_tiny_config(attn_implementation="eager", **kwargs)


def build_pair(config: LlamaConfig, port_class=LlamaForCausalLMForiOS):
    torch.manual_seed(0)
    reference = LlamaForCausalLM(config).eval()
    port = port_class(config, "cpu", disable_embedding_quantization=True).eval()
    load_state_dict_from_ref_model(port, reference)
    return reference, port


@torch.no_grad()
def port_prefill_logits(port, config: LlamaConfig, input_ids: torch.Tensor) -> torch.Tensor:
    seq_len = input_ids.shape[1]
    position_ids = torch.arange(seq_len).unsqueeze(0)
    in_step = torch.tensor([0], dtype=torch.int32)
    key_cache, value_cache = KVCacheHandler.get_kv_cache_from_hf(config)
    mask = _construct_causal_mask(config.max_position_embeddings, seq_len, 0, torch.float32)
    output = port(input_ids, position_ids, in_step, mask, key_cache, value_cache)
    # Apple's iOS code returns (1, 1, S, V) with an lm_head and (1, S, 1, V) with tied embeddings.
    assert output.shape in [(1, 1, seq_len, config.vocab_size), (1, seq_len, 1, config.vocab_size)]
    return output.reshape(1, seq_len, config.vocab_size)


@torch.no_grad()
def compare(config: LlamaConfig, port_class=LlamaForCausalLMForiOS) -> dict:
    reference, port = build_pair(config, port_class)
    torch.manual_seed(1)
    input_ids = torch.randint(1, config.vocab_size, (1, SEQ_LEN))
    expected = reference(input_ids).logits
    actual = port_prefill_logits(port, config, input_ids)
    assert actual.shape == expected.shape, f"{actual.shape} vs {expected.shape}"
    per_position = (actual - expected).abs().amax(dim=-1)[0]
    top1 = (actual.argmax(-1) == expected.argmax(-1)).float().mean().item()
    buckets = {f"{a}-{b - 1}": per_position[a:b].max().item() for a, b in [(0, 64), (64, 256), (256, SEQ_LEN)]}
    result = {"max_abs_diff": per_position.max().item(), "top1": top1, "buckets": buckets}
    print(f"\n{port_class.__name__} tie={config.tie_word_embeddings}: {result}")
    return result


def test_rope_frequencies_match_hugging_face() -> None:
    config = tiny_config()
    _, port = build_pair(config)
    hf_rope = LlamaRotaryEmbedding(config)
    ours = port.extend.rope._inverse_frequencies()
    assert torch.allclose(ours, hf_rope.inv_freq.float(), rtol=0, atol=0)

    # The test config must exercise all three Llama 3 bands.
    wavelengths = 2 * math.pi / LlamaRotaryEmbedding.compute_default_rope_parameters(config)[0]
    original = LLAMA3_ROPE["original_max_position_embeddings"]
    assert (wavelengths < original / LLAMA3_ROPE["high_freq_factor"]).any()
    assert (wavelengths > original / LLAMA3_ROPE["low_freq_factor"]).any()
    medium = (wavelengths >= original / LLAMA3_ROPE["high_freq_factor"]) & (
        wavelengths <= original / LLAMA3_ROPE["low_freq_factor"]
    )
    assert medium.any()


@pytest.mark.parametrize("tie_word_embeddings", [True, False])
def test_port_matches_hugging_face(tie_word_embeddings: bool) -> None:
    result = compare(tiny_config(tie_word_embeddings=tie_word_embeddings))
    assert result["max_abs_diff"] < MAX_ABS_DIFF
    assert result["top1"] == 1.0


def test_default_rope_matches_hugging_face() -> None:
    result = compare(tiny_config(rope={"rope_type": "default", "rope_theta": 500000.0}))
    assert result["max_abs_diff"] < MAX_ABS_DIFF
    assert result["top1"] == 1.0


def test_control_mistral_port_misses_llama3_scaling() -> None:
    """The unmodified Mistral port ignores llama3 scaling. The test must detect that."""
    result = compare(tiny_config(), port_class=MistralForCausalLMForiOS)
    assert result["max_abs_diff"] > CONTROL_MIN_DIFF


@pytest.mark.parametrize("tie_word_embeddings", [True, False])
def test_prompt_and_extend_with_kv_cache(tie_word_embeddings: bool) -> None:
    config = tiny_config(tie_word_embeddings=tie_word_embeddings)
    reference, port = build_pair(config)
    torch.manual_seed(2)
    run_torch_prompt_extend_test_ios(
        port,
        reference,
        precision=torch.float32,
        rtol=MAX_ABS_DIFF,
        atol=MAX_ABS_DIFF,
        strict_compare_numerical=True,
        # Apple's helper needs the transpose only for the lm_head layout (1, 1, S, V).
        use_additional_transpose=not tie_word_embeddings,
    )


@pytest.mark.parametrize(
    "overrides",
    [
        {"rope": {"rope_type": "yarn", "rope_theta": 10000.0, "factor": 4.0}},
        {"attention_bias": True},
        {"mlp_bias": True},
        {"hidden_act": "gelu"},
    ],
)
def test_unsupported_config_is_rejected(overrides: dict) -> None:
    with pytest.raises(ValueError, match="Unsupported Llama config"):
        check_config(tiny_config(**overrides))


def test_register_adds_llama_without_changing_apple_entries() -> None:
    mistral_before = get_model_entry("mistral").ios_class
    register()
    register()
    assert get_model_entry("llama").ios_class is LlamaForCausalLMForiOS
    assert get_model_entry("mistral").ios_class is mistral_before
