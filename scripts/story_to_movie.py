#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Generate a sequence of LTX-2.3 clips and chain them into a short movie."
    )
    parser.add_argument(
        "--story",
        type=Path,
        default=project_root / "config" / "story.example.json",
        help="Path to a story JSON file.",
    )
    parser.add_argument(
        "--ltx-repo",
        type=Path,
        default=project_root / "vendor" / "LTX-2",
        help="Path to the cloned official LTX-2 repository.",
    )
    parser.add_argument(
        "--ltx-python",
        type=Path,
        default=project_root / "vendor" / "LTX-2" / ".venv" / "bin" / "python",
        help="Python executable inside the LTX environment.",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=project_root / "models" / "LTX-2.3" / "ltx-2.3-22b-dev.safetensors",
        help="Path to the official full LTX-2.3 checkpoint.",
    )
    parser.add_argument(
        "--spatial-upscaler-path",
        type=Path,
        default=project_root / "models" / "LTX-2.3" / "ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
        help="Path to the required spatial upscaler checkpoint.",
    )
    parser.add_argument(
        "--distilled-lora-path",
        type=Path,
        default=project_root / "models" / "LTX-2.3" / "ltx-2.3-22b-distilled-lora-384-1.1.safetensors",
        help="Path to the distilled LoRA required by the two-stage pipeline.",
    )
    parser.add_argument(
        "--gemma-root",
        type=Path,
        default=project_root / "models" / "gemma-3-12b-it-qat-q4_0-unquantized",
        help="Path to the Gemma text encoder directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "outputs" / "latest",
        help="Directory for generated clips and metadata.",
    )
    parser.add_argument(
        "--concat",
        action="store_true",
        help="Concatenate scene clips into a single movie if ffmpeg is available.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip scenes whose output files already exist.",
    )
    return parser.parse_args()


def load_story(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if "extends" in data:
        base_path = Path(data["extends"]).expanduser()
        if not base_path.is_absolute():
            base_path = path.parent / base_path
        base_data = load_story(base_path.resolve())
        override = {key: value for key, value in data.items() if key != "extends"}
        data = {**base_data, **override}
    if "scenes" not in data or not isinstance(data["scenes"], list) or not data["scenes"]:
        raise ValueError(f"{path} must contain a non-empty 'scenes' list")
    return data


def log_progress(message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def format_duration(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def ensure_paths(args: argparse.Namespace) -> None:
    required = [
        args.story,
        args.ltx_repo,
        args.ltx_python,
        args.checkpoint_path,
        args.spatial_upscaler_path,
        args.distilled_lora_path,
        args.gemma_root,
    ]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(f"Required path not found: {path}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "frames").mkdir(parents=True, exist_ok=True)


def extract_last_frame(video_path: Path, image_path: Path) -> None:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0:
        capture.release()
        raise RuntimeError(f"No frames found in video: {video_path}")

    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_count - 1)
    ok, frame = capture.read()
    capture.release()
    if not ok or frame is None:
        raise RuntimeError(f"Unable to read last frame from {video_path}")

    image_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(image_path), frame):
        raise RuntimeError(f"Unable to write last frame to {image_path}")


def find_ffmpeg() -> str | None:
    env_ffmpeg = os.environ.get("FFMPEG_BIN")
    if env_ffmpeg:
        return env_ffmpeg

    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg:
        return system_ffmpeg

    try:
        import imageio_ffmpeg
    except ImportError:
        return None
    return imageio_ffmpeg.get_ffmpeg_exe()


def concat_videos(clips: list[Path], output_path: Path) -> bool:
    ffmpeg = find_ffmpeg()
    if ffmpeg is None:
        print("ffmpeg not found; skipping concatenation. Install imageio-ffmpeg or set FFMPEG_BIN.", file=sys.stderr)
        return False

    list_file = output_path.with_suffix(".txt")
    list_file.write_text(
        "".join(f"file '{clip.resolve()}'\n" for clip in clips),
        encoding="utf-8",
    )
    cmd = [
        ffmpeg,
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(list_file),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        str(output_path),
    ]
    subprocess.run(cmd, check=True)
    return True


def build_prompt(story: dict[str, Any], scene: dict[str, Any]) -> str:
    prompt_prefix = str(scene.get("prompt_prefix", story.get("prompt_prefix", ""))).strip()
    prompt_suffix = str(scene.get("prompt_suffix", story.get("prompt_suffix", ""))).strip()
    prompt_parts = [part for part in (prompt_prefix, scene["prompt"], prompt_suffix) if part]
    return " ".join(prompt_parts)


def resolve_image_path(value: str | None) -> Path | None:
    if not value:
        return None
    return Path(value).expanduser().resolve()


def build_command(
    args: argparse.Namespace,
    story: dict[str, Any],
    scene: dict[str, Any],
    index: int,
    output_video: Path,
    conditioning_image: Path | None,
) -> list[str]:
    width = int(scene.get("width", story.get("default_width", 1920)))
    height = int(scene.get("height", story.get("default_height", 1088)))
    num_frames = int(scene.get("num_frames", story.get("default_num_frames", 121)))
    frame_rate = float(scene.get("frame_rate", story.get("default_frame_rate", 24)))
    num_inference_steps = int(scene.get("num_inference_steps", story.get("default_num_inference_steps", 30)))
    seed = int(scene.get("seed", story.get("default_seed", 0) + index))
    pipeline_module = scene.get("pipeline_module", story.get("pipeline_module", "ltx_pipelines.ti2vid_two_stages"))
    quantization = scene.get("quantization", story.get("quantization"))
    enhance_prompt = bool(scene.get("enhance_prompt", story.get("enhance_prompt", False)))
    negative_prompt = scene.get("negative_prompt", story.get("negative_prompt"))
    streaming_prefetch_count = scene.get("streaming_prefetch_count", story.get("streaming_prefetch_count"))
    max_batch_size = int(scene.get("max_batch_size", story.get("max_batch_size", 1)))
    full_prompt = build_prompt(story, scene)

    cmd = [
        str(args.ltx_python),
        "-m",
        pipeline_module,
        "--checkpoint-path",
        str(args.checkpoint_path),
        "--distilled-lora",
        str(args.distilled_lora_path),
        "1.0",
        "--spatial-upsampler-path",
        str(args.spatial_upscaler_path),
        "--gemma-root",
        str(args.gemma_root),
        "--prompt",
        full_prompt,
        "--output-path",
        str(output_video),
        "--seed",
        str(seed),
        "--height",
        str(height),
        "--width",
        str(width),
        "--num-frames",
        str(num_frames),
        "--frame-rate",
        str(frame_rate),
        "--num-inference-steps",
        str(num_inference_steps),
        "--max-batch-size",
        str(max_batch_size),
    ]

    if negative_prompt:
        cmd.extend(["--negative-prompt", str(negative_prompt)])
    if quantization:
        cmd.extend(["--quantization", str(quantization)])
    if streaming_prefetch_count is not None:
        cmd.extend(["--streaming-prefetch-count", str(int(streaming_prefetch_count))])
    if enhance_prompt:
        cmd.append("--enhance-prompt")
    if scene.get("compile", story.get("compile", False)):
        cmd.append("--compile")

    if conditioning_image is not None:
        chain_strength = float(scene.get("chain_strength", story.get("chain_strength", 0.75)))
        cmd.extend(["--image", str(conditioning_image), "0", str(chain_strength)])

    reference_image_mode = str(scene.get("reference_image_mode", story.get("reference_image_mode", "always"))).lower()
    reference_image = resolve_image_path(scene.get("reference_image", story.get("reference_image")))
    should_use_reference_image = reference_image is not None and reference_image_mode != "none"
    if should_use_reference_image and reference_image_mode == "first_scene":
        should_use_reference_image = index == 1
    if should_use_reference_image and reference_image_mode == "when_no_chain":
        should_use_reference_image = conditioning_image is None
    if should_use_reference_image:
        reference_image_strength = float(
            scene.get(
                "reference_image_strength",
                story.get("reference_image_strength", scene.get("start_image_strength", story.get("start_image_strength", 0.95))),
            )
        )
        cmd.extend(["--image", str(reference_image), "0", str(reference_image_strength)])

    start_image = resolve_image_path(scene.get("start_image", story.get("start_image")))
    if start_image is not None:
        start_image_strength = float(scene.get("start_image_strength", story.get("start_image_strength", 0.95)))
        cmd.extend(["--image", str(start_image), "0", str(start_image_strength)])

    return cmd


def main() -> int:
    args = parse_args()
    story = load_story(args.story)
    ensure_paths(args)

    total_scenes = len(story["scenes"])
    run_started_at = time.monotonic()
    completed_scene_durations: list[float] = []

    clips: list[Path] = []
    manifest: dict[str, Any] = {
        "movie_title": story.get("movie_title", "ltx23pro-story-movie"),
        "story_file": str(args.story.resolve()),
        "ltx_repo": str(args.ltx_repo.resolve()),
        "ltx_python": str(args.ltx_python.resolve()),
        "checkpoint_path": str(args.checkpoint_path.resolve()),
        "spatial_upscaler_path": str(args.spatial_upscaler_path.resolve()),
        "distilled_lora_path": str(args.distilled_lora_path.resolve()),
        "gemma_root": str(args.gemma_root.resolve()),
        "clips": [],
    }

    previous_clip: Path | None = None
    for index, scene in enumerate(story["scenes"], start=1):
        scene_id = scene.get("id", f"scene-{index:02d}")
        output_video = args.output_dir / f"{index:02d}_{scene_id}.mp4"

        if args.resume and output_video.exists():
            log_progress(f"Skipping existing scene {index}/{total_scenes}: {output_video.name}")
            clips.append(output_video)
            previous_clip = output_video
            manifest["clips"].append({"scene_id": scene_id, "output": str(output_video.resolve()), "skipped": True})
            continue

        conditioning_image: Path | None = None
        if bool(scene.get("continue_from_previous", True)) and previous_clip and previous_clip.exists():
            conditioning_image = args.output_dir / "frames" / f"{index:02d}_{scene_id}_prev_last_frame.png"
            extract_last_frame(previous_clip, conditioning_image)

        log_progress(f"Starting scene {index}/{total_scenes}: {scene_id}")
        scene_started_at = time.monotonic()
        full_prompt = build_prompt(story, scene)
        cmd = build_command(args, story, scene, index, output_video, conditioning_image)
        log_progress("Running: " + " ".join(cmd))
        subprocess.run(cmd, cwd=args.ltx_repo, check=True)

        scene_elapsed = time.monotonic() - scene_started_at
        completed_scene_durations.append(scene_elapsed)
        clips.append(output_video)
        previous_clip = output_video

        scenes_left = total_scenes - len(completed_scene_durations)
        avg_scene_time = sum(completed_scene_durations) / len(completed_scene_durations)
        eta_seconds = avg_scene_time * scenes_left
        log_progress(
            f"Finished scene {index}/{total_scenes} in {format_duration(scene_elapsed)}. "
            f"Estimated remaining time: {format_duration(eta_seconds)}"
        )

        manifest["clips"].append(
            {
                "scene_id": scene_id,
                "output": str(output_video.resolve()),
                "prompt": scene["prompt"],
                "full_prompt": full_prompt,
                "duration_seconds": scene_elapsed,
                "conditioning_image": str(conditioning_image.resolve()) if conditioning_image else None,
                "reference_image": str(resolve_image_path(scene.get("reference_image", story.get("reference_image"))))
                if scene.get("reference_image", story.get("reference_image"))
                else None,
                "reference_image_mode": scene.get("reference_image_mode", story.get("reference_image_mode", "always")),
                "start_image": str(resolve_image_path(scene.get("start_image", story.get("start_image"))))
                if scene.get("start_image", story.get("start_image"))
                else None,
            }
        )

    manifest_path = args.output_dir / "manifest.json"
    manifest["total_runtime_seconds"] = time.monotonic() - run_started_at
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if args.concat and clips:
        movie_path = args.output_dir / "movie.mp4"
        if concat_videos(clips, movie_path):
            log_progress(f"Concatenated final movie: {movie_path}")

    log_progress(f"All scenes completed in {format_duration(time.monotonic() - run_started_at)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
