"""Run an exported `.aimodel` on the Mac and compare its logits with PyTorch references.

The driver calls the functions the same way as Apple's Swift StaticShapeEngine:
load_embeddings, then gather_embeddings_<q> and extend_<ctx>_<q> per chunk, with
persistent key/value cache states.

Usage:
  ./run.sh python -m coreai_ports.compare_aimodel <bundle_dir> <hf_model_dir> [--tokens 512] [--chunk 16]
"""

import argparse
import asyncio
import json
from pathlib import Path

import numpy as np
import torch
from coreai.runtime import AIModel, NDArray
from transformers import AutoConfig, LlamaForCausalLM

from coreai_ports.llama_ios import LlamaForCausalLMForiOS
from coreai_ports.register import register

MASK_BLOCKED = -40000.0  # Same value as causalMaskSentinel in Apple's Swift runtime.


def interleaved_view(array: NDArray) -> np.ndarray:
    """Writable NumPy view of a rank-5 Core AI state with channel interleave.

    A cache of shape (L, B, C, 1, S) with interleave k is stored as (L, B, C/k, 1, S, k).
    NDArray.numpy() reports per-channel strides for it and reads past the buffer.
    """
    strides = array.strides  # In elements.
    interleave = strides[-1]
    layers, batch, channels, one, seq = array.shape
    if len(array.shape) != 5 or channels % interleave:
        raise ValueError(f"Unexpected state layout: shape {array.shape}, strides {strides}")
    flat = array.numpy()
    item = flat.itemsize
    return np.lib.stride_tricks.as_strided(
        flat,
        shape=(layers, batch, channels // interleave, one, seq, interleave),
        strides=tuple(s * item for s in strides) + (item,),
        writeable=True,
    )


def zeroed_state(descriptor) -> NDArray:
    array = NDArray.from_descriptor(descriptor)
    view = interleaved_view(array)
    view[...] = 0
    if np.any(interleaved_view(array) != 0):
        raise RuntimeError("State buffer did not keep zeros")
    return array


def causal_mask(context: int, chunk: int, step: int, tokens_in_chunk: int) -> np.ndarray:
    mask = np.full((1, context, 1, chunk), MASK_BLOCKED, dtype=np.float16)
    for query in range(tokens_in_chunk):
        mask[0, : min(step + query, context - 1) + 1, 0, query] = 0
    return mask


async def run_aimodel(aimodel_path: Path, input_ids: torch.Tensor, chunk: int) -> tuple[np.ndarray, bool]:
    """Return logits and whether the export quantized the embedding table to int8."""
    model = await AIModel.load(aimodel_path)
    names = set(model.function_names)
    contexts = sorted(int(n.split("_")[1]) for n in names if n.startswith("extend_") and n.endswith(f"_{chunk}"))
    seq_len = input_ids.shape[1]
    context = next((c for c in contexts if c >= seq_len), None)
    if context is None:
        raise ValueError(f"No extend_<ctx>_{chunk} function holds {seq_len} tokens; have {contexts}")

    embeddings = model.load_function("load_embeddings")
    gather = model.load_function(f"gather_embeddings_{chunk}")
    extend = model.load_function(f"extend_{context}_{chunk}")
    table = (await embeddings({}))["embedding_table"]
    int8_embeddings = "int8" in str(embeddings.desc.output_descriptor("embedding_table"))

    state = {name: zeroed_state(extend.desc.state_descriptor(name)) for name in extend.desc.state_names}
    logits = []
    for step in range(0, seq_len, chunk):
        ids = input_ids[0, step : step + chunk].to(torch.int32)
        count = ids.numel()
        padded = torch.zeros(chunk, dtype=torch.int32)
        padded[:count] = ids
        gathered = await gather({"in_new_token_ids": NDArray(data=padded.unsqueeze(0).numpy()), "embedding_table": table})
        inputs = {
            "transformer_input": gathered["gathered_embeddings"],
            "position_ids": NDArray(data=np.arange(step, step + chunk, dtype=np.uint16)[None, :]),
            "in_step": NDArray(data=np.array([step], dtype=np.int32)),
            "causal_mask": NDArray(data=causal_mask(context, chunk, step, count)),
            "embedding_table": table,
        }
        out = await extend(inputs, state=state)
        logits.append(out["out_logits"].numpy().reshape(chunk, -1)[:count])
    return np.concatenate(logits).astype(np.float32)[None], int8_embeddings


@torch.no_grad()
def reference_logits(hf_dir: Path, input_ids: torch.Tensor, rope_type: str) -> np.ndarray:
    config = AutoConfig.from_pretrained(hf_dir)
    if rope_type == "default":
        theta = config.rope_parameters["rope_theta"]
        config.rope_parameters = {"rope_type": "default", "rope_theta": theta}
    model = LlamaForCausalLM.from_pretrained(hf_dir, config=config, torch_dtype=torch.float32).eval()
    return model(input_ids).logits.numpy()


@torch.no_grad()
def port_logits(hf_dir: Path, input_ids: torch.Tensor, context: int, int8_embeddings: bool) -> np.ndarray:
    """Our PyTorch port in float16, with the same embedding type as the export."""
    from coreai_models.primitives.ios.cache import KVCacheHandler

    reference = LlamaForCausalLM.from_pretrained(hf_dir, torch_dtype=torch.float32).eval()
    config = reference.config
    config.max_position_embeddings = context
    port = LlamaForCausalLMForiOS(config, "cpu", disable_embedding_quantization=not int8_embeddings).eval()
    state_dict = dict(reference.state_dict())
    port._mutate_state_dict(state_dict)
    port.load_state_dict(state_dict)
    port = port.to(torch.float16)
    seq_len = input_ids.shape[1]
    key_cache, value_cache = KVCacheHandler.get_kv_cache_from_hf(config, dtype=torch.float16)
    mask = torch.from_numpy(causal_mask(context, seq_len, 0, seq_len))
    out = port(input_ids, torch.arange(seq_len).unsqueeze(0), torch.tensor([0], dtype=torch.int32), mask, key_cache, value_cache)
    return out.reshape(1, seq_len, -1).float().numpy()


def metrics(actual: np.ndarray, expected: np.ndarray, buckets: list[tuple[int, int]]) -> dict:
    per_position = np.abs(actual - expected).max(axis=-1)[0]
    return {
        "max_abs_diff": float(per_position.max()),
        "top1": float((actual.argmax(-1) == expected.argmax(-1)).mean()),
        "buckets": {f"{a}-{b - 1}": float(per_position[a:b].max()) for a, b in buckets if a < len(per_position)},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("bundle", type=Path, help="Export bundle directory that holds the .aimodel")
    parser.add_argument("hf_model", type=Path, help="Hugging Face model directory used for the export")
    parser.add_argument("--tokens", type=int, default=512)
    parser.add_argument("--chunk", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--json", type=Path, help="Write the results to this file")
    args = parser.parse_args()

    register()
    aimodel = next(args.bundle.glob("*.aimodel"))
    vocab_size = AutoConfig.from_pretrained(args.hf_model).vocab_size
    torch.manual_seed(args.seed)
    input_ids = torch.randint(2, vocab_size, (1, args.tokens))
    buckets = [(0, 64), (64, 256), (256, args.tokens)]

    actual, int8_embeddings = asyncio.run(run_aimodel(aimodel, input_ids, args.chunk))
    context = json.loads((args.bundle / "metadata.json").read_text())["language"]["max_context_length"]
    results = {
        "tokens": args.tokens,
        "chunk": args.chunk,
        "int8_embeddings": int8_embeddings,
        "vs_port_float16": metrics(actual, port_logits(args.hf_model, input_ids, context, int8_embeddings), buckets),
        "vs_hf_llama3_float32": metrics(actual, reference_logits(args.hf_model, input_ids, "llama3"), buckets),
        "vs_hf_default_rope_float32": metrics(actual, reference_logits(args.hf_model, input_ids, "default"), buckets),
    }
    text = json.dumps(results, indent=2)
    print(text)
    if args.json:
        args.json.write_text(text + "\n")


if __name__ == "__main__":
    main()
