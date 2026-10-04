"""Download pinned Hugging Face source files. This does not run the model.

Usage: ./run.sh python -m coreai_ports.download <model_id> --revision <sha> --manifest <path>

Weights go to vendor/hf-cache (ignored by Git). The manifest records the revision and files.
"""

import argparse
import json
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

ROOT = Path(__file__).resolve().parents[2]
PATTERNS = ["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja", "LICENSE*", "USE_POLICY*"]
SKIP = ["original/*"]  # Duplicate weights in the publisher's own format.


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_id")
    parser.add_argument("--revision", required=True, help="Full commit SHA to pin")
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    info = HfApi().model_info(args.model_id, revision=args.revision)
    if info.sha != args.revision:
        raise SystemExit(f"Revision mismatch: asked {args.revision}, got {info.sha}")

    folder = Path(
        snapshot_download(
            args.model_id,
            revision=args.revision,
            cache_dir=ROOT / "vendor" / "hf-cache",
            allow_patterns=PATTERNS,
            ignore_patterns=SKIP,
        )
    )
    files = sorted(p for p in folder.rglob("*") if p.is_file())
    if not any(p.suffix == ".safetensors" for p in files):
        raise SystemExit("Download contains no safetensors weights")
    for required in ("config.json", "tokenizer.json", "tokenizer_config.json"):
        if not (folder / required).is_file():
            raise SystemExit(f"Download is missing {required}")

    manifest = {
        "model_id": args.model_id,
        "revision": args.revision,
        "license": getattr(info.card_data, "license", None) if info.card_data else None,
        "gated": info.gated,
        "files": {p.relative_to(folder).as_posix(): p.stat().st_size for p in files},
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Model: {args.model_id} @ {args.revision}")
    print(f"Files: {len(files)}; bytes: {sum(manifest['files'].values())}")
    print(f"Snapshot: {folder}")
    print(f"Manifest: {args.manifest}")


if __name__ == "__main__":
    main()
