"""Export a tiny random Llama 3 model and run the compiled .aimodel on the Mac.

Slow (about 30 s). Needs macOS 27 and the Core AI runtime. Limits are in DECISIONS.md
(2026-10-05); they were set after an exploratory run.
"""

import json
import platform
import subprocess
import sys
from pathlib import Path

import pytest

from coreai_ports.tiny import save_tiny_model

ROOT = Path(__file__).resolve().parents[1]

pytestmark = pytest.mark.skipif(
    platform.system() != "Darwin" or int(platform.mac_ver()[0].split(".")[0] or 0) < 27,
    reason="Core AI runtime needs macOS 27",
)


def run(*args: str) -> None:
    subprocess.run([sys.executable, "-m", *args], check=True, cwd=ROOT, capture_output=True)


@pytest.mark.parametrize("int8_embeddings", [False, True])
def test_compiled_aimodel_matches_hugging_face(tmp_path: Path, int8_embeddings: bool) -> None:
    model_dir = save_tiny_model(tmp_path / "tiny-llama3")
    export_args = [
        "coreai_ports.export", str(model_dir), "--platform", "iOS", "--experimental",
        "--compute-precision", "float16", "--compression", "none",
        "--output-dir", str(tmp_path), "--output-name", "tiny",
    ]
    if not int8_embeddings:
        export_args.append("--disable-embedding-quantization-ios")
    run(*export_args)
    result_path = tmp_path / "result.json"
    run("coreai_ports.compare_aimodel", str(tmp_path / "tiny"), str(model_dir), "--tokens", "512", "--port-reference", "--json", str(result_path))
    result = json.loads(result_path.read_text())

    assert result["int8_embeddings"] == int8_embeddings
    llama3 = result["vs_hf_llama3_float32"]
    control = result["vs_hf_default_rope_float32"]
    # The compiled file holds the Llama 3 RoPE tables, not the default ones.
    assert control["max_abs_diff"] >= 5 * llama3["max_abs_diff"]
    if not int8_embeddings:
        assert llama3["top1"] == 1.0
        assert llama3["max_abs_diff"] < 0.15
