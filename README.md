# coreai-model-ports

Working name. It can change before publication.

This project adds open-source model families to Apple's Core AI exporter for iPhone. Each model is tested before it is listed.

## Status

Nothing is supported yet. The first mission is **Llama 3.2 1B Instruct**. See [missions/01-llama-3.2-1b.md](missions/01-llama-3.2-1b.md).

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

## Supported models

See [SUPPORTED_MODELS.md](SUPPORTED_MODELS.md).

## Requirements

- macOS 27 or later, Xcode 27, the Metal Toolchain, and [uv](https://docs.astral.sh/uv/).
- iOS 27 or later on the phone.
- Apple's [`coreai-models`](https://github.com/apple/coreai-models) at the revision in `DECISIONS.md`.

## License

Not chosen yet. Apple's code is BSD-3-Clause. Keep Apple's notice in any file that copies its code.
