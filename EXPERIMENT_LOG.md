# Experiment log

## 2026-10-04: Inspect Apple's exporter for Llama support

Method: code inspection of `apple/coreai-models` at `52c84ba874b2c57adcede08a671ce96ed1b3f433`. Nothing was run.

Findings:

- iOS LLM presets: 10 models. The registry imports 4 iOS architecture files: Qwen2, Qwen3, Mistral, and OLMo 2. A fifth file, `gemma4_text.py`, exists but is not registered. Llama 3.2 is not listed.
- `models/registry.py` has no `llama` entry. An export of a `llama` model stops with `Unknown model type`.
- SmolLM2 reports `llama` but uses the Qwen2 path through `_model_type_override`.
- No file implements `rope_type: llama3`. Only the macOS files check `rope_scaling` with `is_default_rope_scaling`. The iOS files do not check it.

Interpretation: a Llama 3 model on an existing iOS path would probably lose its RoPE scaling with no error. This is a prediction from code. Mission 1 tests it with a control run.

## 2026-10-04: Llama port accuracy with random weights

Meta's approval for the real weights is pending. This test uses a small random-weight Llama config instead.

### Hypothesis

A subclass of Apple's iOS Mistral port with Llama 3 RoPE frequencies matches Hugging Face's `LlamaForCausalLM`. The unmodified Mistral port does not.

### Prediction

The port passes the limits in `DECISIONS.md`. The control fails them, and its error grows with token position.

### Setup

- Config: 2 layers, hidden size 64, 4 heads, 2 KV heads, head dim 16, vocabulary 256. RoPE: Llama 3.2 values (theta 500000, `llama3`, factor 32, low 1, high 4, original 8192). These frequencies cover all three Llama 3 bands. `initializer_range` 0.2.
- float32, CPU, `USE_HF_IMPL=true`. One 1000-token prompt. Same random weights in both models.
- Command: `./run.sh pytest tests -s`.

### Result

12 tests passed.

| Model | Max logit difference | Top-1 match | Max difference at positions 0–63 / 64–255 / 256–999 |
|---|---|---|---|
| Llama port, tied embeddings | 1.16e-5 | 100% | 6.7e-6 / 7.2e-6 / 1.2e-5 |
| Llama port, untied | 1.43e-5 | 100% | 5.2e-6 / 8.1e-6 / 1.4e-5 |
| Llama port, default RoPE | 1.59e-5 | 100% | — |
| Control: unmodified Mistral port | 5.99 | 63.3% | 0.36 / 1.82 / 5.99 |

- RoPE frequencies are identical to Hugging Face's `LlamaRotaryEmbedding` (zero difference).
- Apple's prompt + extend helper passes for tied and untied models with tolerance 1e-3.
- Unsupported configs (`yarn`, attention bias, MLP bias, GELU) stop with a clear error.

### Interpretation

The port is numerically correct on random weights. The control proves the test detects missing Llama 3 scaling. Without the port, the error is silent and grows with position. This does not test real weights, float16, compression, or the phone.

### Findings during the test

- Apple's iOS code returns logits as `(1, 1, S, V)` with an lm_head and `(1, S, 1, V)` with tied embeddings. Apple's test helper handles only the first layout. Our tests handle both. Llama 3.2 1B uses tied embeddings, like Qwen3 0.6B, which runs on the iPhone 15.
- Transformers warns that `original_max_position_embeddings` (8192) exceeds `max_position_embeddings`. The `llama3` function uses the original value, so the frequencies are correct. The same warning will appear at 4096-token export.

## 2026-10-04: Tiny Llama through Apple's exporter

### Setup

The same random config, saved locally with a 512-token context. Command: `./run.sh python -m coreai_ports.export vendor/tiny-llama3 --platform iOS --experimental --compute-precision float16 --compression none --max-context-length 512`.

### Result

- Apple's exporter found the registered `llama` port. Export, compilation, and save of the `.aimodel` succeeded (412 KB).
- The command then failed at the tokenizer step, because the tiny model has no tokenizer files. The real model has them.
- With the default iOS compression, palettization failed on `layers.0.self_attn.q_proj.weight` (64×64). The cause is not known. It may be the tiny shape. Test again with real weights.
- Models without an Apple registry preset need `--experimental` and `--compute-precision`.

### Interpretation

The port traces and compiles for iOS. The compiled graph has not been compared numerically with PyTorch, and it has not run on the phone.

### Next question

Does the compiled `.aimodel` give the same logits as PyTorch on the Mac? Apple's `run_compare_coreai` helper may answer this before any phone test.
