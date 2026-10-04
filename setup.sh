#!/bin/bash
# Clone Apple's coreai-models at the pinned revision and install its locked environment.
set -euo pipefail

root="$(cd "$(dirname "$0")" && pwd)"
apple_repo="${COREAI_MODELS_DIR:-$root/vendor/coreai-models}"
apple_url="https://github.com/apple/coreai-models.git"
apple_revision="52c84ba874b2c57adcede08a671ce96ed1b3f433"

if ! command -v uv >/dev/null; then
    echo "uv is required: https://docs.astral.sh/uv/"
    exit 1
fi
if [[ ! -d "$apple_repo/.git" ]]; then
    git clone --filter=blob:none "$apple_url" "$apple_repo"
fi
git -C "$apple_repo" fetch --quiet origin "$apple_revision"
git -C "$apple_repo" checkout --quiet --detach "$apple_revision"
echo "Apple coreai-models revision: $(git -C "$apple_repo" rev-parse HEAD)"

# The default dependency group includes pytest, which the accuracy tests need.
cd "$apple_repo"
uv sync --frozen
echo "Ready: $apple_repo"
