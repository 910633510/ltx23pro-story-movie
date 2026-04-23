#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

from naruto_vid2vid_lib import (
    load_config,
    load_json,
    manifest_path_for_config,
    parse_frame_rate,
    resolve_path,
    run,
    write_json,
)


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Render one Naruto vid2vid chunk with the official LTX-2 IC-LoRA pipeline.")
    parser.add_argument(
        "--config",
        type=Path,
        default=project_root / "config" / "naruto_379_vid2vid_4k_3min.json",
        help="Path to the Naruto vid2vid config JSON.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Chunk manifest path. Defaults to the path derived from the config.",
    )
    parser.add_argument(
        "--chunk-index",
        type=int,
        default=None,
        help="Zero-based chunk index. Intended for Slurm array tasks.",
    )
    parser.add_argument(
        "--chunk-id",
        type=str,
        default=None,
        help="Explicit chunk id, for example chunk-0001.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-render the chunk even if the output file already exists.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved LTX command without executing it.",
    )
    return parser.parse_args()


def select_chunk(manifest: dict[str, object], chunk_index: int | None, chunk_id: str | None) -> dict[str, object]:
    chunks = manifest["chunks"]
    if chunk_id is not None:
        for chunk in chunks:
            if chunk["chunk_id"] == chunk_id:
                return chunk
        raise KeyError(f"Chunk id not found: {chunk_id}")
    if chunk_index is None:
        raise ValueError("Provide either --chunk-index or --chunk-id")
    if chunk_index < 0 or chunk_index >= len(chunks):
        raise IndexError(f"Chunk index {chunk_index} is out of range for {len(chunks)} chunks")
    return chunks[chunk_index]


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    manifest_path = args.manifest or manifest_path_for_config(config, args.config)
    manifest = load_json(manifest_path)
    chunk = select_chunk(manifest, args.chunk_index, args.chunk_id)

    control_mode = chunk["control_mode"]
    if control_mode not in ("pose", "canny"):
        raise ValueError(f"Chunk {chunk['chunk_id']} has unresolved control_mode={control_mode!r}")

    control_cfg = config["pose_control"] if control_mode == "pose" else config["canny_control"]
    control_video = Path(chunk["pose_control_video"] if control_mode == "pose" else chunk["canny_control_video"])
    lora_path = resolve_path(control_cfg["lora_path"], args.config.parent)
    distilled_checkpoint_path = resolve_path(config["distilled_checkpoint_path"], args.config.parent)
    spatial_upsampler_path = resolve_path(config["spatial_upsampler_path"], args.config.parent)
    gemma_root = resolve_path(config["gemma_root"], args.config.parent)
    ltx_repo_dir = resolve_path(config["ltx_repo_dir"], args.config.parent)
    ltx_python = resolve_path(config["ltx_python"], args.config.parent)
    output_path = Path(chunk["rendered_video"])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and not args.force:
        print(f"Skipping existing rendered chunk: {output_path}", flush=True)
        return 0

    for required_path in (control_video, lora_path, distilled_checkpoint_path, spatial_upsampler_path, gemma_root, ltx_python):
        if not required_path.exists():
            raise FileNotFoundError(f"Required path not found: {required_path}")

    width = int(config["working_width"])
    height = int(config["working_height"])
    _, fps_text = parse_frame_rate(config["frame_rate"])
    num_frames = int(chunk["num_frames"])
    seed = int(config.get("seed_base", 379000)) + int(chunk["chunk_index"])
    prompt = str(config["prompt"]).strip()

    cmd = [
        str(ltx_python),
        "-m",
        "ltx_pipelines.ic_lora",
        "--distilled-checkpoint-path",
        str(distilled_checkpoint_path),
        "--spatial-upsampler-path",
        str(spatial_upsampler_path),
        "--gemma-root",
        str(gemma_root),
        "--prompt",
        prompt,
        "--output-path",
        str(output_path),
        "--seed",
        str(seed),
        "--height",
        str(height),
        "--width",
        str(width),
        "--num-frames",
        str(num_frames),
        "--frame-rate",
        fps_text,
        "--video-conditioning",
        str(control_video),
        str(float(control_cfg.get("conditioning_strength", 1.0))),
        "--lora",
        str(lora_path),
        str(float(control_cfg.get("lora_strength", 1.0))),
        "--max-batch-size",
        str(int(config.get("max_batch_size", 1))),
    ]

    if config.get("enhance_prompt", False):
        cmd.append("--enhance-prompt")
    if config.get("streaming_prefetch_count") is not None:
        cmd.extend(["--streaming-prefetch-count", str(int(config["streaming_prefetch_count"]))])
    if config.get("compile", False):
        cmd.append("--compile")
    if config.get("quantization"):
        cmd.extend(["--quantization", str(config["quantization"])])

    command_record = {
        "chunk_id": chunk["chunk_id"],
        "control_mode": control_mode,
        "control_video": str(control_video),
        "lora_path": str(lora_path),
        "command": cmd,
        "negative_prompt_note": config.get(
            "negative_prompt",
            "",
        ),
    }
    write_json(output_path.with_suffix(".json"), command_record)

    if args.dry_run:
        print(" ".join(cmd), flush=True)
        return 0

    env = os.environ.copy()
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    run(cmd, cwd=ltx_repo_dir, env=env)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
