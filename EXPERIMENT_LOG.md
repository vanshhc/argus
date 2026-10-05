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

The same random config, saved locally with a 512-token context. Command: `./run.sh python -m argus_kit.export vendor/tiny-llama3 --platform iOS --experimental --compute-precision float16 --compression none --max-context-length 512`.

### Result

- Apple's exporter found the registered `llama` port. Export, compilation, and save of the `.aimodel` succeeded (412 KB).
- The command then failed at the tokenizer step, because the tiny model has no tokenizer files. The real model has them.
- With the default iOS compression, palettization failed on `layers.0.self_attn.q_proj.weight` (64×64). The cause is not known. It may be the tiny shape. Test again with real weights.
- Models without an Apple registry preset need `--experimental` and `--compute-precision`.

### Interpretation

The port traces and compiles for iOS. The compiled graph has not been compared numerically with PyTorch, and it has not run on the phone.

### Next question

Does the compiled `.aimodel` give the same logits as PyTorch on the Mac? Apple's `run_compare_coreai` helper may answer this before any phone test.

## 2026-10-05: Compiled `.aimodel` on the Mac

### Hypothesis

The exported `.aimodel` holds the Llama 3 RoPE tables. On the Mac Core AI runtime it gives the same logits as PyTorch.

### Prediction

The pass limits in `DECISIONS.md` (2026-10-05), set before the run.

### Setup

- Tiny random Llama 3 model with a word-level tokenizer (`argus_kit.tiny`). The full export now completes.
- Export: iOS, float16, no compression, 512-token context. Default `int8` embeddings.
- `argus_kit.compare_aimodel` runs the file like Apple's Swift `StaticShapeEngine`: 16-token chunks of `extend_512_16`, persistent KV states, `uint16` positions, mask -40000. 512 random tokens.

### Result: first run (`int8` embeddings)

| Comparison | Max abs diff | Top-1 | Limit | Pass |
|---|---|---|---|---|
| vs our PyTorch port, float16, `int8` embeddings | 0.232 | 97.3% | < 0.05, ≥ 99% | **No** |
| vs Hugging Face float32, Llama 3 RoPE | 0.659 | 94.3% | < 0.25, ≥ 98% | **No** |
| Control: vs Hugging Face float32, default RoPE | 6.82 | 77.5% | ≥ 5× the Llama 3 diff | Yes (10.4×) |

Two limits failed. The control passed: the error against default RoPE grows with position (0.64 / 1.71 / 6.82), so the file holds the Llama 3 tables.

### Diagnosis (exploratory, after the failure)

1. Our PyTorch port in float16 with `int8` embeddings is already 0.560 from Hugging Face float32 (top-1 93.9%). The gap exists before compilation.
2. Export with float embeddings (`--disable-embedding-quantization-ios`), all else equal:

| Comparison | Max abs diff | Top-1 |
|---|---|---|
| vs our PyTorch port, float16 | 0.071 | 99.2% |
| vs Hugging Face float32, Llama 3 RoPE | **0.062** | **100%** |
| Control: vs default RoPE | 6.10 | 77.3% |

3. Float32 iOS export fails in Apple's pipeline (`fused_dequant_gather_reshape: dtype f16 vs f32`; with float embeddings, a `slice_update` type error). Apple's own Mistral port fails the same way. Float32 is not available as a control.
4. The 60 "Incompatible element type for ANE" messages also appear on first load of Apple's own Mistral port. A second load shows none (specialization cache).

### Interpretation

- The port and Apple's compiler are correct for this model. With float embeddings, the compiled file matches Hugging Face float32 to 0.062 with 100% top-1.
- `int8` per-tensor embedding quantization causes most of the error. Llama 3.2 1B ties its embeddings, so the `int8` table also produces the output logits.
- PyTorch float16 on CPU is a poor reference. It is further from float32 than the compiled file.
- My first limits did not account for `int8` embeddings and were wrong for this model. I did not change them after the run. New limits for the regression test are in `DECISIONS.md`.
- These are random weights with `initializer_range` 0.2. The size of the `int8` effect on real Llama weights is unknown.

### Next question

How much does `int8` embedding quantization change Llama 3.2 1B outputs with real weights, and how much memory does float16 embeddings cost?

## 2026-10-05: Real-weight PyTorch accuracy, Llama 3.2 1B Instruct

### Hypothesis

With the real weights, the port matches Hugging Face in float32 at all positions. Apple's unmodified Mistral port does not.

### Prediction

The limits in `DECISIONS.md` (2026-10-05, real-weight test), set before the run.

### Setup

- Weights: `meta-llama/Llama-3.2-1B-Instruct` @ `9213176726f574b556790deb65791e0c5aa438b6`. `model.safetensors` SHA-256 `1ff795ff…538f` matches the Hugging Face file ID. Manifest: `models/llama-3.2-1b-instruct/source-model.json`.
- The real `config.json` matches the mission plan: 16 layers, hidden 2048, heads 32/8/64, `llama3` RoPE factor 32, tied embeddings, vocabulary 128,256, no bias, SiLU.
- Text: the model's `LICENSE.txt` and `USE_POLICY.md`, 3,055 tokens.
- float32, CPU, `USE_HF_IMPL=true`. 256-token chunks with a KV cache on both sides. Shared weights.
- Command: `./run.sh python -m argus_kit.compare_torch <snapshot> --tokens 4096 --chunk 256 --control`. Results: `results/llama-3.2-1b-instruct/torch-float32.json`.

### Result

| Model | Max abs logit diff | Top-1 | Max diff at 0–63 / 64–255 / 256–1023 / 1024–3054 | Pass |
|---|---|---|---|---|
| Llama port | **1.69e-4** | **100%** (0 of 3,055 differ) | 1.7e-4 / 8.7e-5 / 1.0e-4 / 1.3e-4 | Yes |
| Control: Mistral port | 7.06 | 90.9% (279 differ) | 0.48 / 1.77 / 5.46 / 7.06 | — |

Control ratio: 7.06 / 1.69e-4 ≈ 41,700× (limit ≥ 10×). Run time 298 s; peak resident memory 8.3 GB.

### Interpretation

All three pre-set limits passed. The port matches Hugging Face Llama 3.2 1B in float32 at every position. Without the port, the existing iOS path changes about 1 in 11 predicted tokens on this text, and the error grows with position. This is silent: nothing in Apple's iOS path reports it.

This does not test float16, `int8` embeddings, compression, the compiled file, or the phone.

### Problems fixed during this run

- Memory: building the port allocated its own random weights before `assign=True` replaced them. Peak use reached 10 GB on the 16 GB Mac and it swapped. Fix: build the port on the `meta` device.
- Speed: PyTorch sent some of the port's 1×1 convolutions to NNPACK, about 1.4 s per call on this CPU. With NNPACK off, a 2-layer forward pass dropped from 5.92 s to 0.12 s with identical output (difference 0.0). `compare_torch` turns NNPACK off.

## 2026-10-05: Llama 3.2 1B exports and Mac load (no accuracy result)

### Exports (succeeded)

Float16 compute, 4096-token context, no weight compression. Offline export by model ID (`HF_HUB_OFFLINE=1`; `refs/main` points to the pinned revision).

| Export | `main.mlirb` | Time | Peak memory |
|---|---|---|---|
| `int8` embeddings | 2,220,329,407 bytes | 113 s | 6.6 GB |
| float16 embeddings | 2,482,991,436 bytes | 113 s | 6.9 GB |

The difference is 262,662,029 bytes. This matches the predicted 128,256 × 2,048 bytes for the embedding table. Both bundles record `meta-llama/Llama-3.2-1B-Instruct` and include the tokenizer and chat template.

### Mac load (failed)

1. Default compute units, `int8` export: after about 20 minutes, the Neural Engine reported `Program load failed — no memory (transient; retry under lower memory pressure)`. MPSGraph then aborted the process (exit 134). Peak resident memory 12.2 GB. No token ran. The float16-embedding run was stopped because it would load the same way.
2. Compute-unit options on the tiny export: default and GPU-preferred load. CPU-preferred and CPU-only fail with `AIModelError error 1`. The runtime has no GPU-only option; GPU-preferred still allows the Neural Engine.
3. GPU-preferred, `int8` export: no output after 2 hours (background time limit). The process never finished `AIModel.load`. The first Python log line came 17 minutes after start, so the Mac may have slept or been under heavy load. The cause of the stall is not known.

### Interpretation

- The export works for the full model. The uncompressed file (2.2 GB) did not load on this 16 GB Mac. The Neural Engine ran out of memory.
- The iPhone 15 has 6 GB. An uncompressed export is very unlikely to load there. Weight compression is probably required, not optional.
- The `int8` vs float16 embedding comparison has no result. Repeat it on compressed exports.

### Next question

Does a compressed export (Apple's 4-bit palettization or 6-bit preset) load on the Mac, and how much accuracy does it cost against Hugging Face float32?

## 2026-10-05: Compression track — 4-bit `group32` on Llama 3.2 1B

This is the compression track, separate from the conversion mission (user decision, 2026-10-05).

### Fix needed first

Every compressed export failed with `Centroid calculation failed`. The real error was `Ninja is required to load C++ extensions`: Apple's palettizer compiles a C++ k-means helper, and `run.sh` did not put the environment's `bin/` on `PATH`. Fixed in `run.sh`.

### Setup

Apple's `4bit_weight_palettized_group32` preset. For iOS, palettization uses no calibration data: plain k-means on the weights, 16 values shared by each group of 32 output channels. The embedding table is not palettized. Two exports differ only in the table. Mac comparison with default compute units, 3,055 tokens of license text.

### Result

| Export | File | Loads (default units) | Perplexity (reference 8.58) | Rise | Top-1 | Max diff |
|---|---|---|---|---|---|---|
| `int8` table | 0.76 GB | Yes: 62 s load, 217 s run | 14.59 | +70.1% | 64.4% | 15.7 |
| float16 table | 1.02 GB | Yes: 64 s load, 215 s run | 14.45 | +68.5% | 64.3% | 16.0 |

- Perplexity rule (≤ 10%): **fails** for both.
- Control (≥ 5×): **fails** (ratio 1.0). The compression error is larger than the RoPE effect, so this control cannot separate them. The RoPE fix was proven on the uncompressed model.
- Embedding rule: top-1 differs by 0.1 points, so keep the `int8` table.
- A tiny random model lost even more (top-1 36%); random weights have no structure to compress.

### Interpretation

The compressed files load and run on the Mac, but plain 4-bit `group32` costs too much quality for this 1B model. The embedding table is not the cause. Apple's own small-model presets avoid plain 4-bit: Qwen3 0.6B uses mixed 4/8-bit, Qwen3 1.7B and SmolLM2 1.7B use 6-bit.

### Next question (compression track)

How much of the loss do `group8` and 6-bit recover, and at what size?
