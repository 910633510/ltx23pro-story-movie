#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import snapshot_download

LTX_REPO_ID = "Lightricks/LTX-2.3"
GEMMA_REPO_ID = "google/gemma-3-12b-it-qat-q4_0-unquantized"

OPEN_MODEL_FILES = [
    "ltx-2.3-22b-dev.safetensors",
    "ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
    "ltx-2.3-22b-distilled-lora-384-1.1.safetensors",
]

OPTIONAL_TEMPORAL_FILE = "ltx-2.3-temporal-upscaler-x2-1.0.safetensors"


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Download the official LTX-2.3 full-model assets required by this project."
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=project_root / "models" / "LTX-2.3",
        help="Directory where the LTX model bundle will be stored.",
    )
    parser.add_argument(
        "--gemma-dir",
        type=Path,
        default=project_root / "models" / "gemma-3-12b-it-qat-q4_0-unquantized",
        help="Directory where the Gemma text encoder files will be stored.",
    )
    parser.add_argument(
        "--include-temporal-upscaler",
        action="store_true",
        help="Also download the optional temporal upscaler file.",
    )
    return parser.parse_args()


def env_token() -> str | None:
    for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
        value = os.environ.get(key)
        if value:
            return value
    return None


def main() -> int:
    args = parse_args()
    token = env_token()

    args.model_dir.mkdir(parents=True, exist_ok=True)
    args.gemma_dir.mkdir(parents=True, exist_ok=True)

    allow_patterns = list(OPEN_MODEL_FILES)
    if args.include_temporal_upscaler:
        allow_patterns.append(OPTIONAL_TEMPORAL_FILE)

    print(f"Downloading {LTX_REPO_ID} assets into {args.model_dir}", flush=True)
    snapshot_download(
        repo_id=LTX_REPO_ID,
        repo_type="model",
        local_dir=str(args.model_dir),
        allow_patterns=allow_patterns,
        token=token,
        resume_download=True,
    )

    if token is None:
        print(
            "HF token not found in environment. Skipping Gemma download.\n"
            "Set HF_TOKEN after accepting the Gemma terms, then rerun the bootstrap.",
            flush=True,
        )
        return 0

    print(f"Downloading gated Gemma assets into {args.gemma_dir}", flush=True)
    snapshot_download(
        repo_id=GEMMA_REPO_ID,
        repo_type="model",
        local_dir=str(args.gemma_dir),
        token=token,
        resume_download=True,
    )
    print("LTX-2.3 asset download complete.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
