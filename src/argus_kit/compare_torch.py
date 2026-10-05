"""Compare the PyTorch port with Hugging Face on real weights, in chunks with a KV cache.

Both models share one copy of the weights. Only per-position metrics are kept, so
full-vocabulary logits for long inputs never sit in memory at once.

Usage:
  ./run.sh python -m argus_kit.compare_torch <hf_model_dir> [--tokens 4096] [--chunk 256] [--control]
"""

import argparse
import gc
import json
import os
from pathlib import Path

# Apple's Hugging Face parity path. It is read when the port's modules are built.
os.environ["USE_HF_IMPL"] = "true"

import torch  # noqa: E402
from transformers import DynamicCache, LlamaForCausalLM  # noqa: E402

from coreai_models.models.ios.mistral import MistralForCausalLMForiOS  # noqa: E402
from coreai_models.primitives.ios.cache import KVCacheHandler  # noqa: E402
from argus_kit.data import TEXT_FILES, text_tokens  # noqa: E402
from argus_kit.llama_ios import LlamaForCausalLMForiOS  # noqa: E402

def causal_mask(context: int, chunk: int, offset: int) -> torch.Tensor:
    mask = torch.zeros((1, context, 1, chunk), dtype=torch.float32)
    for i in range(chunk):
        mask[:, offset + i + 1 :, :, i] = float("-inf")
    return mask


# NNPACK takes about 1.4 s per 1x1 convolution in the port on Apple CPUs. The default
# path gives identical results about 50 times faster.
torch.backends.nnpack.set_flags(False)


@torch.no_grad()
def compare(hf_dir: Path, input_ids: torch.Tensor, chunk: int, port_class) -> dict:
    reference = LlamaForCausalLM.from_pretrained(hf_dir, dtype=torch.float32).eval()
    config = reference.config
    seq_len = input_ids.shape[1]
    config.max_position_embeddings = seq_len

    # Build on the meta device so the port allocates no weights of its own. RoPE tables
    # are computed on the CPU inside RoPECache. assign=True shares the reference weights.
    with torch.device("meta"):
        port = port_class(config, "cpu", disable_embedding_quantization=True).eval()
    state_dict = dict(reference.state_dict())
    port._mutate_state_dict(state_dict)
    port.load_state_dict(state_dict, assign=True, strict=True)
    on_meta = [name for name, t in [*port.named_parameters(), *port.named_buffers()] if t.is_meta]
    if on_meta:
        raise RuntimeError(f"Port tensors left on the meta device: {on_meta[:5]}")

    key_cache, value_cache = KVCacheHandler.get_kv_cache_from_hf(config)
    hf_cache = DynamicCache(config=config)
    diffs, matches = [], []
    for start in range(0, seq_len, chunk):
        ids = input_ids[:, start : start + chunk]
        count = ids.shape[1]
        expected = reference(input_ids=ids, past_key_values=hf_cache, use_cache=True).logits[0]
        actual = port(
            ids,
            torch.arange(start, start + count).unsqueeze(0),
            torch.tensor([start], dtype=torch.int32),
            causal_mask(seq_len, count, start),
            key_cache,
            value_cache,
        ).reshape(count, -1)
        diffs.append((actual - expected).abs().amax(dim=-1))
        matches.append(actual.argmax(-1) == expected.argmax(-1))
        print(f"  {port_class.__name__}: positions {start}-{start + count - 1} max diff {diffs[-1].max():.2e}", flush=True)

    per_position = torch.cat(diffs)
    top1 = torch.cat(matches).float()
    buckets = [(0, 64), (64, 256), (256, 1024), (1024, seq_len)]
    result = {
        "max_abs_diff": per_position.max().item(),
        "top1": top1.mean().item(),
        "top1_mismatch_positions": torch.nonzero(top1 == 0).flatten().tolist(),
        "buckets": {f"{a}-{b - 1}": per_position[a:b].max().item() for a, b in buckets if a < seq_len},
    }
    del reference, port, state_dict, hf_cache
    gc.collect()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("hf_model", type=Path)
    parser.add_argument("--tokens", type=int, default=4096)
    parser.add_argument("--chunk", type=int, default=256)
    parser.add_argument("--control", action="store_true", help="Also run the unmodified Mistral port")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    input_ids = text_tokens(args.hf_model, args.tokens)
    print(f"Tokens: {input_ids.shape[1]} from {', '.join(TEXT_FILES)}", flush=True)
    results = {"tokens": input_ids.shape[1], "chunk": args.chunk}
    results["port"] = compare(args.hf_model, input_ids, args.chunk, LlamaForCausalLMForiOS)
    if args.control:
        results["control_mistral_port"] = compare(args.hf_model, input_ids, args.chunk, MistralForCausalLMForiOS)
    text = json.dumps(results, indent=2)
    print(text)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text + "\n")


if __name__ == "__main__":
    main()
