# Mission 1 — Llama 3.2 1B Instruct on iPhone

Model: `meta-llama/Llama-3.2-1B-Instruct`. Phone: iPhone 15, iOS 27.0.1.

## Problem

Apple's exporter has no `llama` entry for iOS. An export of Llama 3.2 stops with `Unknown model type 'llama'`.

Apple's iOS code also has no Llama 3 RoPE scaling. The iOS files do not check `rope_scaling`. If Llama 3.2 used the existing Mistral path, the exporter would probably ignore the scaling with no error. The output would then be wrong. This is not tested yet.

## What you will learn

- How a decoder model becomes a fixed-shape graph for the phone.
- How RoPE works, and how Llama 3 changes its frequencies.
- How to prove that a port matches the original model.

## Hypothesis

Llama 3.2 1B differs from Apple's iOS Mistral port mainly in RoPE scaling. A port with Llama 3 RoPE scaling will match Hugging Face outputs on the Mac and run on the iPhone 15.

## Prediction

- Without the scaling, logits will differ from Hugging Face. The difference will grow with token position.
- With the scaling, float32 logits will match Hugging Face closely, and the top-1 tokens will be the same.
- The 4-bit export will fit and run on the iPhone 15. Speed and memory are unknown.

These are predictions, not results.

## Known config (from Meta's model card; verify from `config.json`)

| Field | Expected value |
|---|---|
| Layers / hidden size | 16 / 2048 |
| Attention heads / KV heads / head dim | 32 / 8 / 64 |
| RoPE | theta 500000, `rope_type: llama3`, factor 32 |
| Tied input and output embeddings | Yes |
| Vocabulary | 128,256 |

## Steps

1. **User action.** Accept Meta's license on the Hugging Face model page. Log in with `hf auth login`. The assistant does not handle the token.
2. Download the pinned source revision. Record it in `models/llama/source-model.json`.
3. Write `ports/llama_ios.py`. Start from Apple's iOS Mistral port. Add Llama 3 RoPE scaling. Reject unsupported config values with a clear error.
4. Register `llama` with Apple's exporter at run time.
5. Run the Mac accuracy test in float32: Hugging Face against the port, same prompts, positions up to 4096. Record max logit difference and top-1 agreement. Run it once without the scaling as a control.
6. Export for iOS with float16 compute. Use no compression first, then 4-bit palettization.
7. Load the export in the Argus test app on the iPhone 15. Record preparation time, response, and memory.
8. Add the result to `SUPPORTED_MODELS.md`.

## Success

- The Mac accuracy test passes. The pass limits are set before the first run and recorded in `DECISIONS.md`.
- The iPhone gives a correct answer to a fixed prompt.
- The control run shows that the scaling changes the result. This proves the test can catch the error.

## Out of scope

- Other models.
- A configurable decoder for many families. Decide this after mission 1.
- Publication on GitHub.
