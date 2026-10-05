"""Run an exported `.aimodel` on the Mac and compare its logits with PyTorch references.

The driver calls the functions the same way as Apple's Swift StaticShapeEngine:
load_embeddings, then gather_embeddings_<q> and extend_<ctx>_<q> per chunk, with
persistent key/value cache states.

References are compared chunk by chunk with a KV cache, one model in memory at a time.

Usage:
  ./run.sh python -m argus_kit.compare_aimodel <bundle_dir> <hf_model_dir> [--text] [--tokens 512]
"""

import argparse
import asyncio
import json
import time
from pathlib import Path

import numpy as np
import torch
from coreai.runtime import AIModel, ComputeUnitKind, NDArray, SpecializationOptions
from transformers import AutoConfig, DynamicCache, LlamaForCausalLM

from argus_kit.data import text_tokens
from argus_kit.llama_ios import LlamaForCausalLMForiOS
from argus_kit.register import register

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


async def run_aimodel(aimodel_path: Path, input_ids: torch.Tensor, chunk: int, compute_units: str = "default") -> tuple[np.ndarray, dict]:
    """Return float16 logits (positions, vocab) and run facts."""
    started = time.perf_counter()
    # "gpu" prefers the GPU. The runtime has no GPU-only option, and its CPU backend
    # cannot load iOS exports (AIModelError 1).
    options = (
        SpecializationOptions.from_preferred_compute_unit_kind(ComputeUnitKind.gpu())
        if compute_units == "gpu"
        else SpecializationOptions.default()
    )
    model = await AIModel.load(aimodel_path, options)
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
    loaded = time.perf_counter()

    state = {name: zeroed_state(extend.desc.state_descriptor(name)) for name in extend.desc.state_names}
    logits = None
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
        out = (await extend(inputs, state=state))["out_logits"].numpy().reshape(chunk, -1)[:count]
        if logits is None:
            logits = np.empty((seq_len, out.shape[-1]), dtype=np.float16)
        logits[step : step + count] = out
    finished = time.perf_counter()
    facts = {
        "int8_embeddings": int8_embeddings,
        "function": f"extend_{context}_{chunk}",
        "compute_units": compute_units,
        "mac_load_seconds": round(loaded - started, 2),
        "mac_run_seconds": round(finished - loaded, 2),
    }
    return logits, facts


def perplexity(logits: torch.Tensor, input_ids: torch.Tensor) -> float:
    """exp(mean negative log-likelihood of each real next token). Lower is better."""
    targets = input_ids[0, 1 : logits.shape[0] + 1]
    log_probs = torch.log_softmax(logits[: targets.shape[0]].float(), dim=-1)
    return torch.exp(-log_probs.gather(1, targets.unsqueeze(1)).mean()).item()


def metrics(diffs: torch.Tensor, matches: torch.Tensor, buckets: list[tuple[int, int]]) -> dict:
    return {
        "max_abs_diff": diffs.max().item(),
        "top1": matches.float().mean().item(),
        "top1_mismatches": int((~matches).sum().item()),
        "buckets": {f"{a}-{b - 1}": diffs[a:b].max().item() for a, b in buckets if a < len(diffs)},
    }


@torch.no_grad()
def compare_reference(hf_dir: Path, input_ids: torch.Tensor, actual: np.ndarray, rope_type: str, buckets, chunk: int = 256) -> dict:
    """Hugging Face float32, run in chunks with a KV cache."""
    config = AutoConfig.from_pretrained(hf_dir)
    if rope_type == "default":
        theta = config.rope_parameters["rope_theta"]
        config.rope_parameters = {"rope_type": "default", "rope_theta": theta}
    model = LlamaForCausalLM.from_pretrained(hf_dir, config=config, dtype=torch.float32).eval()
    cache = DynamicCache(config=model.config)
    diffs, matches, nll_sum, nll_count = [], [], 0.0, 0
    targets = input_ids[0, 1:]
    for start in range(0, input_ids.shape[1], chunk):
        expected = model(input_ids=input_ids[:, start : start + chunk], past_key_values=cache, use_cache=True).logits[0]
        got = torch.from_numpy(actual[start : start + expected.shape[0]]).float()
        diffs.append((got - expected).abs().amax(dim=-1))
        matches.append(got.argmax(-1) == expected.argmax(-1))
        step_targets = targets[start : start + expected.shape[0]]
        log_probs = torch.log_softmax(expected[: step_targets.shape[0]], dim=-1)
        nll_sum -= log_probs.gather(1, step_targets.unsqueeze(1)).sum().item()
        nll_count += step_targets.shape[0]
    del model, cache
    result = metrics(torch.cat(diffs), torch.cat(matches), buckets)
    result["reference_perplexity"] = float(np.exp(nll_sum / nll_count))
    return result


@torch.no_grad()
def compare_port(hf_dir: Path, input_ids: torch.Tensor, actual: np.ndarray, context: int, int8_embeddings: bool, buckets) -> dict:
    """Our PyTorch port in float16, with the same embedding type as the export. Small models only."""
    from coreai_models.primitives.ios.cache import KVCacheHandler

    reference = LlamaForCausalLM.from_pretrained(hf_dir, dtype=torch.float32).eval()
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
    expected = out.reshape(seq_len, -1).float()
    got = torch.from_numpy(actual).float()
    return metrics((got - expected).abs().amax(dim=-1), got.argmax(-1) == expected.argmax(-1), buckets)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("bundle", type=Path, help="Export bundle directory that holds the .aimodel")
    parser.add_argument("hf_model", type=Path, help="Hugging Face model directory used for the export")
    parser.add_argument("--tokens", type=int, default=512)
    parser.add_argument("--chunk", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--text", action="store_true", help="Use the model's own license text, not random tokens")
    parser.add_argument("--port-reference", action="store_true", help="Also compare with the float16 PyTorch port")
    parser.add_argument("--compute-units", choices=["default", "gpu"], default="default")
    parser.add_argument("--json", type=Path, help="Write the results to this file")
    args = parser.parse_args()

    register()
    torch.backends.nnpack.set_flags(False)  # NNPACK is very slow for the port's 1x1 convolutions.
    aimodel = next(args.bundle.glob("*.aimodel"))
    if args.text:
        input_ids = text_tokens(args.hf_model, args.tokens)
    else:
        torch.manual_seed(args.seed)
        input_ids = torch.randint(2, AutoConfig.from_pretrained(args.hf_model).vocab_size, (1, args.tokens))
    seq_len = input_ids.shape[1]
    buckets = [(0, 64), (64, 256), (256, 1024), (1024, seq_len)]

    actual, facts = asyncio.run(run_aimodel(aimodel, input_ids, args.chunk, args.compute_units))
    print(f"Ran {facts['function']} on {seq_len} tokens: {facts}", flush=True)
    results = {"tokens": seq_len, "chunk": args.chunk, "input": "text" if args.text else "random", **facts}
    if args.text:
        results["aimodel_perplexity"] = perplexity(torch.from_numpy(actual), input_ids)
    if args.port_reference:
        context = json.loads((args.bundle / "metadata.json").read_text())["language"]["max_context_length"]
        results["vs_port_float16"] = compare_port(args.hf_model, input_ids, actual, context, facts["int8_embeddings"], buckets)
    results["vs_hf_llama3_float32"] = compare_reference(args.hf_model, input_ids, actual, "llama3", buckets)
    results["vs_hf_default_rope_float32"] = compare_reference(args.hf_model, input_ids, actual, "default", buckets)
    text = json.dumps(results, indent=2)
    print(text)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text + "\n")


if __name__ == "__main__":
    main()
