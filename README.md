# Argus

Argus adds open-source models to Apple's Core AI on iPhone. In Greek myth, Argus was the giant with a hundred eyes who never stopped watching. This project watches every number: each model is checked against the original before it is listed.

It adds new model families to Apple's Core AI exporter. Each model is tested before it is listed.

## Status

Nothing is supported yet. The Llama port matches Hugging Face Llama 3.2 1B in float32 on real weights. Export and phone tests are next. The first mission is **Llama 3.2 1B Instruct**. See [missions/01-llama-3.2-1b.md](missions/01-llama-3.2-1b.md).

## What this project is

- New iOS architecture code for model families that Apple's exporter does not support.
- An accuracy test on the Mac. It compares each port with the original Hugging Face model.
- Measured results on real iPhones.

## What this project is not

- It is not a new inference engine. Apple's Core AI runs the model.
- It is not a replacement for Apple's exporter. It uses Apple's exporter, compression, and compiler as a dependency.
- It does not support "any model." Each architecture needs code and tests.

## How it works

Apple's exporter converts a model in three steps:

1. A hand-written PyTorch version of the architecture, built from Apple's iOS primitives.
2. The original Hugging Face weights, loaded into that version.
3. Compression and compilation to `.aimodel`.

This project supplies step 1 for new families. It registers each new family with Apple's exporter at run time. It does not fork Apple's code.

## Rules for claims

- List a model as supported only after it passes the Mac accuracy test and runs on a real iPhone.
- Give the phone, iOS version, and export settings for each result.
- Do not publish model weights or exports. Users download weights under each model's license.

## Use

```sh
bash setup.sh                 # pinned Apple clone and locked environment in vendor/
./run.sh pytest tests -s      # accuracy tests with random weights; no downloads (about 45 s)
./run.sh python -m argus_kit.export <hf_model_id> --platform iOS --experimental --compute-precision float16
```

Compare an exported bundle with Hugging Face on the Mac:

```sh
./run.sh python -m argus_kit.compare_aimodel <bundle_dir> <hf_model_dir> --tokens 512
```

`argus_kit.export` is Apple's exporter with this project's ports registered. Models without an Apple preset need `--experimental`.

## Supported models

See [SUPPORTED_MODELS.md](SUPPORTED_MODELS.md).

## Requirements

- macOS 27 or later, Xcode 27, the Metal Toolchain, and [uv](https://docs.astral.sh/uv/).
- iOS 27 or later on the phone.
- Apple's [`coreai-models`](https://github.com/apple/coreai-models) at the revision in `DECISIONS.md`.

## License

BSD-3-Clause. See [LICENSE](LICENSE). Apple's `coreai-models` uses the same license. This project uses Apple's code as a dependency. One function in `src/argus_kit/llama_ios.py` is adapted from Apple's RoPE cache and keeps Apple's notice.

Model weights are not part of this project. Each model has its own license. For example, Llama 3.2 uses the Llama 3.2 Community License.
