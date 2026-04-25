#!/usr/bin/env python3
"""Run the official HunyuanVideo-1.5 I2V smoke test from a repo-local JSON config."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_path(value: str | None) -> Path | None:
    if value is None:
        return None
    expanded = Path(os.path.expandvars(os.path.expanduser(value)))
    if expanded.is_absolute():
        return expanded
    return PROJECT_ROOT / expanded


def bool_arg(value: bool) -> str:
    return "true" if value else "false"


def command_to_shell(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def required_model_paths(model_path: Path, sr_enabled: bool) -> list[Path]:
    required = [
        model_path / "config.json",
        model_path / "scheduler" / "scheduler_config.json",
        model_path / "vae" / "diffusion_pytorch_model.safetensors",
        model_path / "transformer" / "480p_i2v_step_distilled" / "diffusion_pytorch_model.safetensors",
        model_path / "text_encoder" / "llm" / "config.json",
        model_path / "text_encoder" / "byt5-small" / "config.json",
        model_path / "text_encoder" / "Glyph-SDXL-v2" / "checkpoints" / "byt5_model.pt",
        model_path / "vision_encoder" / "siglip" / "image_encoder" / "model.safetensors",
    ]
    if sr_enabled:
        required.extend(
            [
                model_path / "transformer" / "720p_sr_distilled" / "diffusion_pytorch_model.safetensors",
                model_path / "upsampler" / "720p_sr_distilled" / "diffusion_pytorch_model.safetensors",
            ]
        )
    return required


def validate_paths(config: dict[str, Any], repo_dir: Path, model_path: Path, image_path: Path, output_path: Path) -> None:
    generate_py = repo_dir / "generate.py"
    required = [generate_py, image_path]
    required.extend(required_model_paths(model_path, bool(config.get("sr", True))))
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        message = "Missing required Hunyuan files:\n" + "\n".join(f"- {path}" for path in missing)
        message += "\nRun scripts/bootstrap_hunyuan_hpc.sh on the HPC login node first."
        raise FileNotFoundError(message)
    output_path.parent.mkdir(parents=True, exist_ok=True)


def build_command(config: dict[str, Any], repo_dir: Path, model_path: Path, image_path: Path, output_path: Path) -> list[str]:
    generate_py = repo_dir / "generate.py"
    nproc_per_node = int(config.get("nproc_per_node", 1))
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        f"--nproc_per_node={nproc_per_node}",
        str(generate_py),
        "--prompt",
        str(config["prompt"]),
        "--negative_prompt",
        str(config.get("negative_prompt", "")),
        "--image_path",
        str(image_path),
        "--resolution",
        str(config.get("resolution", "480p")),
        "--aspect_ratio",
        str(config.get("aspect_ratio", "16:9")),
        "--seed",
        str(int(config.get("seed", 123))),
        "--num_inference_steps",
        str(int(config.get("num_inference_steps", 12))),
        "--video_length",
        str(int(config.get("video_length", 121))),
        "--rewrite",
        bool_arg(bool(config.get("rewrite", False))),
        "--cfg_distilled",
        bool_arg(bool(config.get("cfg_distilled", False))),
        "--enable_step_distill",
        bool_arg(bool(config.get("enable_step_distill", True))),
        "--sparse_attn",
        bool_arg(bool(config.get("sparse_attn", False))),
        "--use_sageattn",
        bool_arg(bool(config.get("use_sageattn", False))),
        "--enable_cache",
        bool_arg(bool(config.get("enable_cache", False))),
        "--cache_type",
        str(config.get("cache_type", "deepcache")),
        "--sr",
        bool_arg(bool(config.get("sr", True))),
        "--save_pre_sr_video",
        bool_arg(bool(config.get("save_pre_sr_video", True))),
        "--offloading",
        bool_arg(bool(config.get("offloading", True))),
        "--overlap_group_offloading",
        bool_arg(bool(config.get("overlap_group_offloading", True))),
        "--dtype",
        str(config.get("dtype", "bf16")),
        "--save_generation_config",
        "true",
        "--output_path",
        str(output_path),
        "--model_path",
        str(model_path),
    ]

    group_offloading = config.get("group_offloading")
    if group_offloading is not None:
        command.extend(["--group_offloading", bool_arg(bool(group_offloading))])

    for optional in ("checkpoint_path", "lora_path"):
        value = config.get(optional)
        if value:
            resolved = resolve_path(str(value))
            command.extend([f"--{optional}", str(resolved)])

    return command


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a HunyuanVideo-1.5 I2V smoke test.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config" / "hunyuan_i2v_smoke_test.json"))
    parser.add_argument("--dry-run", action="store_true", help="Validate config shape and print the official command.")
    args = parser.parse_args()

    config_path = resolve_path(args.config)
    if config_path is None:
        raise ValueError("Config path is required.")
    config = load_config(config_path)

    repo_dir = resolve_path(str(config.get("hunyuan_repo_dir", "vendor/HunyuanVideo-1.5")))
    model_path = resolve_path(str(config.get("model_path", "models/HunyuanVideo-1.5")))
    image_path = resolve_path(str(config["image_path"]))
    output_path = resolve_path(str(config.get("output_path", "outputs/hunyuan_i2v_smoke/hunyuan_i2v_smoke.mp4")))
    assert repo_dir is not None and model_path is not None and image_path is not None and output_path is not None

    command = build_command(config, repo_dir, model_path, image_path, output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command_file = output_path.parent / "hunyuan_i2v_smoke_command.sh"
    command_file.write_text(command_to_shell(command) + "\n", encoding="utf-8")

    print(f"Config: {config_path}")
    print(f"Hunyuan repo: {repo_dir}")
    print(f"Model path: {model_path}")
    print(f"Reference image: {image_path}")
    print(f"Output path: {output_path}")
    print(f"Command file: {command_file}")
    print(command_to_shell(command))

    if args.dry_run:
        return 0

    validate_paths(config, repo_dir, model_path, image_path, output_path)
    env = os.environ.copy()
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:128")
    subprocess.run(command, cwd=repo_dir, env=env, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
