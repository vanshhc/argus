# Experiment log

## 2026-10-04: Inspect Apple's exporter for Llama support

Method: code inspection of `apple/coreai-models` at `52c84ba874b2c57adcede08a671ce96ed1b3f433`. Nothing was run.

Findings:

- iOS LLM presets: 10 models. The registry imports 4 iOS architecture files: Qwen2, Qwen3, Mistral, and OLMo 2. A fifth file, `gemma4_text.py`, exists but is not registered. Llama 3.2 is not listed.
- `models/registry.py` has no `llama` entry. An export of a `llama` model stops with `Unknown model type`.
- SmolLM2 reports `llama` but uses the Qwen2 path through `_model_type_override`.
- No file implements `rope_type: llama3`. Only the macOS files check `rope_scaling` with `is_default_rope_scaling`. The iOS files do not check it.

Interpretation: a Llama 3 model on an existing iOS path would probably lose its RoPE scaling with no error. This is a prediction from code. Mission 1 tests it with a control run.
