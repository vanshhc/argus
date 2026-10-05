#!/bin/bash
# Run a command in Apple's environment with this project and Apple's test helpers importable.
# Examples:
#   ./run.sh pytest tests
#   ./run.sh python -m argus_kit.export meta-llama/Llama-3.2-1B-Instruct --platform iOS
set -euo pipefail

root="$(cd "$(dirname "$0")" && pwd)"
apple_repo="${COREAI_MODELS_DIR:-$root/vendor/coreai-models}"
apple_revision="52c84ba874b2c57adcede08a671ce96ed1b3f433"

if [[ ! -x "$apple_repo/.venv/bin/python" ]]; then
    echo "Apple environment not found at $apple_repo. Run: bash setup.sh"
    exit 1
fi
if [[ "$(git -C "$apple_repo" rev-parse HEAD)" != "$apple_revision" ]]; then
    echo "Apple coreai-models is not at the pinned revision $apple_revision."
    exit 1
fi

export PYTHONPATH="$root/src:$apple_repo/python${PYTHONPATH:+:$PYTHONPATH}"
# Tools in the environment must be on PATH. Compression builds a C++ helper with ninja.
export PATH="$apple_repo/.venv/bin:$PATH"
command="$1"
shift
exec "$apple_repo/.venv/bin/$command" "$@"
