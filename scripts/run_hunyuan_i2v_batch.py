#!/usr/bin/env python3
"""Run a time-bounded batch of official HunyuanVideo-1.5 I2V generations."""

from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from run_hunyuan_i2v_smoke import (
    PROJECT_ROOT,
    build_command,
    command_to_shell,
    load_config,
    resolve_path,
    validate_paths,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp_path.replace(path)


def format_prompt_id(prompt_id: str) -> str:
    allowed = []
    for char in prompt_id.lower():
        if char.isalnum() or char in ("-", "_"):
            allowed.append(char)
        elif char.isspace():
            allowed.append("-")
    cleaned = "".join(allowed).strip("-_")
    return cleaned or "clip"


def render_prompt(base_config: dict[str, Any], prompt_item: str | dict[str, Any]) -> tuple[str, str, dict[str, Any]]:
    overrides: dict[str, Any] = {}
    if isinstance(prompt_item, str):
        prompt_id = format_prompt_id(prompt_item[:48])
        text = prompt_item
    else:
        prompt_id = format_prompt_id(str(prompt_item.get("id", "clip")))
        text = str(prompt_item.get("text", ""))
        overrides = {key: value for key, value in prompt_item.items() if key not in {"id", "text"}}

    prefix = str(base_config.get("prompt_prefix", ""))
    suffix = str(base_config.get("prompt_suffix", ""))
    return prompt_id, f"{prefix}{text}{suffix}", overrides


def output_path_for(config: dict[str, Any], index: int, prompt_id: str) -> Path:
    output_dir = resolve_path(str(config.get("output_dir", "outputs/hunyuan_i2v_12h_longrun")))
    assert output_dir is not None
    template = str(config.get("output_name_template", "{index:03d}_{prompt_id}.mp4"))
    name = template.format(index=index + 1, prompt_id=prompt_id)
    return output_dir / name


def average_completed_seconds(entries: list[dict[str, Any]]) -> float | None:
    durations = [
        float(entry["elapsed_seconds"])
        for entry in entries
        if entry.get("status") == "completed" and isinstance(entry.get("elapsed_seconds"), (int, float))
    ]
    if not durations:
        return None
    return sum(durations) / len(durations)


def load_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"created_at": utc_now(), "updated_at": utc_now(), "clips": []}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def completed_outputs(manifest: dict[str, Any]) -> set[str]:
    completed: set[str] = set()
    for entry in manifest.get("clips", []):
        if entry.get("status") == "completed":
            completed.add(str(entry.get("output_path", "")))
    return completed


def should_stop_for_time(start_time: float, config: dict[str, Any], manifest: dict[str, Any], clip_index: int) -> bool:
    target_run_seconds = float(config.get("target_run_seconds", 41400))
    stop_buffer_seconds = float(config.get("stop_buffer_seconds", 1800))
    elapsed = time.monotonic() - start_time
    if elapsed >= target_run_seconds - stop_buffer_seconds:
        return True

    avg = average_completed_seconds(manifest.get("clips", []))
    if avg is None:
        return False

    projected = elapsed + avg + stop_buffer_seconds
    if projected > target_run_seconds:
        print(
            f"Stopping before clip {clip_index + 1}: elapsed={elapsed:.1f}s, "
            f"avg_clip={avg:.1f}s, buffer={stop_buffer_seconds:.1f}s, target={target_run_seconds:.1f}s",
            flush=True,
        )
        return True
    return False


def build_clip_config(base_config: dict[str, Any], prompt: str, seed: int, output_path: Path, overrides: dict[str, Any]) -> dict[str, Any]:
    clip_config = copy.deepcopy(base_config)
    for unused in ("prompts", "prompt_prefix", "prompt_suffix", "output_dir", "output_name_template", "manifest_path", "command_dir"):
        clip_config.pop(unused, None)
    clip_config.update(overrides)
    clip_config["prompt"] = prompt
    clip_config["seed"] = seed
    clip_config["output_path"] = str(output_path)
    return clip_config


def run_batch(config_path: Path, dry_run: bool, resume: bool) -> int:
    config = load_config(config_path)
    prompts = config.get("prompts")
    if not isinstance(prompts, list) or not prompts:
        raise ValueError("Batch config must contain a non-empty prompts list.")

    repo_dir = resolve_path(str(config.get("hunyuan_repo_dir", "vendor/HunyuanVideo-1.5")))
    model_path = resolve_path(str(config.get("model_path", "models/HunyuanVideo-1.5")))
    image_path = resolve_path(str(config["image_path"]))
    manifest_path = resolve_path(str(config.get("manifest_path", "outputs/hunyuan_i2v_12h_longrun/batch_manifest.json")))
    command_dir = resolve_path(str(config.get("command_dir", "outputs/hunyuan_i2v_12h_longrun/commands")))
    assert repo_dir is not None and model_path is not None and image_path is not None
    assert manifest_path is not None and command_dir is not None

    max_clips = int(config.get("max_clips", len(prompts)))
    seed_start = int(config.get("seed", 123))
    seed_stride = int(config.get("seed_stride", 1))
    effective_resume = resume or bool(config.get("resume", False))
    manifest = load_manifest(manifest_path)
    already_completed = completed_outputs(manifest)
    start_time = time.monotonic()

    first_prompt_id, first_prompt, first_overrides = render_prompt(config, prompts[0])
    first_output = output_path_for(config, 0, first_prompt_id)
    first_config = build_clip_config(config, first_prompt, seed_start, first_output, first_overrides)
    if not dry_run:
        validate_paths(first_config, repo_dir, model_path, image_path, first_output)

    print(f"Config: {config_path}")
    print(f"Hunyuan repo: {repo_dir}")
    print(f"Model path: {model_path}")
    print(f"Reference image: {image_path}")
    print(f"Manifest: {manifest_path}")
    print(f"Max clips: {max_clips}")
    print(f"Target seconds: {config.get('target_run_seconds', 41400)}")

    env = os.environ.copy()
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:128")

    for index in range(max_clips):
        if should_stop_for_time(start_time, config, manifest, index):
            break

        prompt_item = prompts[index % len(prompts)]
        prompt_id, prompt, overrides = render_prompt(config, prompt_item)
        output_path = output_path_for(config, index, prompt_id)
        seed = int(overrides.get("seed", seed_start + index * seed_stride))
        clip_config = build_clip_config(config, prompt, seed, output_path, overrides)
        command = build_command(clip_config, repo_dir, model_path, image_path, output_path)
        command_path = command_dir / f"{index + 1:03d}_{prompt_id}.sh"
        command_path.parent.mkdir(parents=True, exist_ok=True)
        command_path.write_text(command_to_shell(command) + "\n", encoding="utf-8")

        if dry_run:
            print(f"[dry-run] clip {index + 1:03d} seed={seed} output={output_path}")
            print(command_to_shell(command))
            if index >= min(max_clips, 3) - 1:
                print("[dry-run] showing first 3 commands only")
                break
            continue

        if effective_resume and (str(output_path) in already_completed or (output_path.exists() and output_path.stat().st_size > 0)):
            print(f"Skipping completed clip {index + 1:03d}: {output_path}", flush=True)
            continue

        entry = {
            "clip_index": index + 1,
            "prompt_id": prompt_id,
            "seed": seed,
            "output_path": str(output_path),
            "command_path": str(command_path),
            "status": "running",
            "started_at": utc_now(),
        }
        manifest.setdefault("clips", []).append(entry)
        manifest["updated_at"] = utc_now()
        write_json(manifest_path, manifest)

        print(f"Starting clip {index + 1:03d}: prompt_id={prompt_id} seed={seed}", flush=True)
        clip_start = time.monotonic()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            subprocess.run(command, cwd=repo_dir, env=env, check=True)
        except subprocess.CalledProcessError as exc:
            entry["status"] = "failed"
            entry["returncode"] = exc.returncode
            entry["finished_at"] = utc_now()
            entry["elapsed_seconds"] = time.monotonic() - clip_start
            manifest["updated_at"] = utc_now()
            write_json(manifest_path, manifest)
            raise

        entry["status"] = "completed"
        entry["finished_at"] = utc_now()
        entry["elapsed_seconds"] = time.monotonic() - clip_start
        manifest["updated_at"] = utc_now()
        write_json(manifest_path, manifest)
        print(f"Completed clip {index + 1:03d} in {entry['elapsed_seconds']:.1f}s", flush=True)

    manifest["finished_at"] = utc_now()
    manifest["total_elapsed_seconds"] = time.monotonic() - start_time
    manifest["updated_at"] = utc_now()
    write_json(manifest_path, manifest)
    print(f"Batch finished in {manifest['total_elapsed_seconds']:.1f}s")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run HunyuanVideo-1.5 I2V clips until the configured time budget is nearly used.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config" / "hunyuan_i2v_12h_longrun.json"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Skip clips already completed in the manifest or present on disk.")
    args = parser.parse_args()

    config_path = resolve_path(args.config)
    if config_path is None:
        raise ValueError("Config path is required.")
    return run_batch(config_path, args.dry_run, args.resume)


if __name__ == "__main__":
    raise SystemExit(main())
