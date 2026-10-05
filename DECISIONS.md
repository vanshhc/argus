# Decisions

## User-selected (2026-10-04)

- A separate repository from the iPhone app. The app is the product; this is infrastructure. (The app was called Argus until 2026-10-05; see below.)
- Goal: support open-source models beyond Apple's iOS list, each one tested.
- First model: Llama 3.2 1B Instruct.

## Provisional choices

These are the assistant's choices. The user can change them.

| Choice | Reason | Trade-off |
|---|---|---|
| Working name `coreai-model-ports` (superseded 2026-10-05: **Argus**, user-selected) | Describes the work; no Apple trademark as the first word | Can change before publication |
| Use Apple's `coreai-models` as a pinned dependency, not a fork | Less code to maintain; Apple keeps compression and compilation | Apple's internal API can change between revisions |
| Register new families at run time | No edits to Apple's files | Uses Apple's internal registry, not a public plugin API |
| Start the Llama port from Apple's iOS Mistral port | Closest existing architecture | Must check every difference, not only RoPE |
| Mac accuracy test before any phone test | Finds wrong output early and cheaply | Needs the full model in float32 on the Mac |
| No license yet (superseded 2026-10-05: BSD-3-Clause, user-selected) | User had not chosen one | Do not publish until chosen |

## Pinned dependency

`apple/coreai-models` revision `52c84ba874b2c57adcede08a671ce96ed1b3f433`. This is the revision tested on the iPhone 15 in the study workspace.

## 2026-10-04: Llama port design and test limits

Problem: Meta has not approved weight access yet. The port code and its accuracy test do not need the real weights.

Choice:

- `LlamaForCausalLMForiOS` subclasses Apple's iOS Mistral port. Llama and Mistral share layer names and block structure. Only the RoPE module changes.
- The new RoPE module gets its frequencies from Hugging Face's own `ROPE_INIT_FUNCTIONS`. This uses the same formula as the reference. It does not copy it.
- Supported `rope_type` values: `default` and `llama3`. Any other value, attention bias, MLP bias, or a non-SiLU activation stops with a clear error.
- Test with a small random-weight Llama config. Its RoPE frequencies cover all three Llama 3 bands (high, medium, low).

Trade-off: the subclass depends on Apple's internal `MistralExtend.rope` attribute. The pinned revision protects us; a test fails if the attribute moves.

Pass limits, set before the first run (float32, CPU, `USE_HF_IMPL=true`):

| Check | Pass |
|---|---|
| Port vs Hugging Face, max absolute logit difference, all positions | < 1e-3 |
| Top-1 token agreement, all positions | 100% |
| Apple's prompt + extend helper (KV cache path) | Passes |
| Control: Llama config through the unmodified Mistral port | Max difference > 1e-2, so the test can detect missing scaling |

## 2026-10-05: Compare the exported `.aimodel` on the Mac

Problem: the PyTorch tests do not prove that the exporter put the Llama 3 RoPE tables into the compiled file.

Choice: load the exported `.aimodel` with `coreai.runtime.AIModel` on the Mac. Run it the same way as Apple's Swift `StaticShapeEngine`: `load_embeddings`, then `gather_embeddings_16` and `extend_<ctx>_16` per 16-token chunk, with persistent `key_cache`/`value_cache` states, `uint16` positions, the `int32` step, and a mask with -40000 for blocked entries. Compare with three references on the same tokens.

Trade-off: the Mac runtime is not the iPhone runtime. Compute units and specialization can differ. This test finds export and compile errors; it does not replace the phone test.

Pass limits, set before the first run (tiny random model, float16 export, `int8` embeddings, 512 tokens):

| Check | Pass |
|---|---|
| `.aimodel` vs our PyTorch port in float16 with the same `int8` embeddings | Max abs logit difference < 0.05; top-1 ≥ 99% |
| `.aimodel` vs Hugging Face float32 (Llama 3 RoPE) | Max abs logit difference < 0.25; top-1 ≥ 98% |
| Control: `.aimodel` vs Hugging Face float32 with default RoPE | Max difference ≥ 5 × the difference to the Llama 3 reference |

### Result and new limits (2026-10-05)

The first two limits failed. The cause was `int8` embedding quantization, not the port. See `EXPERIMENT_LOG.md`. The original limits stay recorded as failed.

Regression limits for `tests/test_aimodel_mac.py`. **These were set after the exploratory run**, with margin over the observed values:

| Export | Check | Pass |
|---|---|---|
| float16, float embeddings | vs Hugging Face float32: top-1 | 100% |
| float16, float embeddings | vs Hugging Face float32: max abs diff | < 0.15 (observed 0.062) |
| Both | Control: default-RoPE diff vs Llama 3 diff | ≥ 5× |

Open choice for the real model: `int8` embeddings (Apple's default) or float16 embeddings. For Llama 3.2 1B the table has 128,256 × 2,048 values: about 263 MB in `int8`, about 525 MB in float16. Measure both on real weights before choosing.

## 2026-10-05: Real-weight PyTorch accuracy test (Llama 3.2 1B Instruct)

Choice: `argus_kit.compare_torch` runs Hugging Face and the port in float32 on the CPU, in 256-token chunks with a KV cache on both sides. Both models share one copy of the weights (`load_state_dict(assign=True)`). The text is the model's own downloaded `LICENSE.txt` and `USE_POLICY.md`, so the test needs no other data. Only per-position metrics are kept.

Trade-off: license text is not typical chat text. It is real language, and it reaches about 3,000 positions.

Pass limits, set before the first run:

| Check | Pass |
|---|---|
| Port vs Hugging Face, max abs logit difference, all positions | < 0.05 |
| Top-1 agreement, all positions | ≥ 99.9% |
| Control: Mistral port (no Llama 3 scaling) with the same weights | Max diff ≥ 10 × the port's max diff |

## 2026-10-05: Llama 3.2 1B exports — `int8` vs float16 embeddings

Setup: two iOS exports. The only difference is the embedding table: `int8` (Apple's default) or float16 (`--disable-embedding-quantization-ios`). Both use float16 compute, a 4096-token context, and no weight compression. Compression is a separate test. Each file runs on the Mac with `compare_aimodel --text` (3,055 tokens), against Hugging Face float32.

Limits, set before the first run:

| Export | Check | Pass |
|---|---|---|
| float16 embeddings | Top-1 vs Hugging Face float32 | ≥ 99% |
| float16 embeddings | Max abs logit diff | < 1.0 |
| Both | Control: default-RoPE diff vs Llama 3 diff | ≥ 5× |

Decision rule, set before the first run: keep `int8` embeddings if their top-1 is within 1 percentage point of the float16-embedding export. Otherwise use float16 embeddings and accept about 263 MB more. The phone test can still change this choice through memory limits.

## 2026-10-05: Publication plan (user-selected)

- This toolkit gets its own **public** GitHub repository. The iPhone app gets a separate **private** repository.
- Before the first public push: choose a license, confirm the name, and check that no weights, exports, or local diagnostics are in Git.

- License (2026-10-05, user-selected): BSD-3-Clause, the same as Apple's `coreai-models`. Copyright line uses the git name `vanshhc`; the user can change it.

## 2026-10-05: Name (user-selected)

- Project name: **Argus**. GitHub repository: `argus`. Python package: `argus_kit`, because `argus` is taken on PyPI.
- Before this, Argus was the name of the user's iPhone app. The app gets a new name later. The installed app on the phone still shows Argus until then.
- Local folder: `/Users/Shared/argus` (was `/Users/Shared/coreai-model-ports`).

## 2026-10-05: Compressed exports (Llama 3.2 1B)

Problem: the uncompressed files (2.2–2.5 GB) did not load on the Mac's Neural Engine. The iPhone 15 has 6 GB.

Choice: Apple's iOS default compression, `4bit_weight_palettized_group32`. Each layer weight becomes a 4-bit index into a table of 16 values, with one table per group of 32 output channels. Apple's preset leaves the embedding table out of palettization, so two exports differ only in the table: `int8` or float16. Everything else as before (float16 compute, 4096-token context).

New metric: perplexity on the 3,055-token license text, the exponent of the mean negative log-likelihood of each real next token. Lower is better. It measures quality, not only agreement with the reference.

Limits and rules, set before the first run:

| Check | Rule |
|---|---|
| File loads and runs on the Mac (default compute units) | Required |
| Perplexity increase vs Hugging Face float32 | ≤ 10%: ready for the phone test. Above 10%: try `group8` next |
| Control: default-RoPE diff vs Llama 3 diff | ≥ 5× |
| `int8` vs float16 embeddings | Keep `int8` if its top-1 is within 1 percentage point of float16 (rule from 2026-10-05) |

Predicted sizes, before the run: about 0.75 GB (`int8` table) and 1.0 GB (float16 table).
