#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from naruto_vid2vid_lib import (
    count_extracted_frames,
    ensure_dir,
    load_config,
    parse_frame_rate,
    probe_video,
    require_ffmpeg,
    resolve_path,
    run,
    write_json,
)


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Extract the first 3 minutes of Naruto-379 into a 4K vid2vid workspace.")
    parser.add_argument(
        "--config",
        type=Path,
        default=project_root / "config" / "naruto_379_vid2vid_4k_3min.json",
        help="Path to the Naruto vid2vid config JSON.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-extract the segment, audio, and frame sequence even if they already exist.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    ffmpeg = require_ffmpeg()

    source_video = resolve_path(config["source_video"], args.config.parent)
    workspace_root = ensure_dir(resolve_path(config["workspace_root"], args.config.parent))
    source_frames_dir = ensure_dir(workspace_root / "source_frames")
    segment_path = workspace_root / "source_segment.mp4"
    audio_path = workspace_root / "source_audio.m4a"

    start_seconds = float(config["segment_start_seconds"])
    end_seconds = float(config["segment_end_seconds"])
    duration_seconds = end_seconds - start_seconds
    if duration_seconds <= 0:
        raise ValueError("segment_end_seconds must be greater than segment_start_seconds")

    target_width = int(config["working_width"])
    target_height = int(config["working_height"])
    _, fps_text = parse_frame_rate(config["frame_rate"])

    metadata = probe_video(source_video)
    if metadata["width"] != target_width or metadata["height"] != target_height:
        raise ValueError(
            f"Source video is {metadata['width']}x{metadata['height']}, expected {target_width}x{target_height} for native 4K."
        )

    existing_frames = count_extracted_frames(source_frames_dir)
    if not args.force and segment_path.exists() and audio_path.exists() and existing_frames > 0:
        print(f"Using existing extracted workspace at {workspace_root}", flush=True)
    else:
        if args.force:
            for frame_file in source_frames_dir.glob("frame-*.png"):
                frame_file.unlink()

        run(
            [
                ffmpeg,
                "-y",
                "-ss",
                f"{start_seconds:.3f}",
                "-i",
                str(source_video),
                "-t",
                f"{duration_seconds:.3f}",
                "-c:v",
                "libx264",
                "-preset",
                "medium",
                "-crf",
                "18",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                str(segment_path),
            ]
        )

        run(
            [
                ffmpeg,
                "-y",
                "-ss",
                f"{start_seconds:.3f}",
                "-i",
                str(source_video),
                "-t",
                f"{duration_seconds:.3f}",
                "-vn",
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                str(audio_path),
            ]
        )

        run(
            [
                ffmpeg,
                "-y",
                "-ss",
                f"{start_seconds:.3f}",
                "-i",
                str(source_video),
                "-t",
                f"{duration_seconds:.3f}",
                "-vf",
                f"fps={fps_text},scale={target_width}:{target_height}:flags=lanczos",
                "-start_number",
                "1",
                str(source_frames_dir / "frame-%06d.png"),
            ]
        )

    extracted_frame_count = count_extracted_frames(source_frames_dir)
    if extracted_frame_count <= 0:
        raise RuntimeError("No PNG frames were extracted.")

    segment_metadata = {
        "source_video": str(source_video),
        "segment_path": str(segment_path),
        "audio_path": str(audio_path),
        "source_frames_dir": str(source_frames_dir),
        "segment_start_seconds": start_seconds,
        "segment_end_seconds": end_seconds,
        "duration_seconds": duration_seconds,
        "frame_rate": fps_text,
        "working_width": target_width,
        "working_height": target_height,
        "extracted_frame_count": extracted_frame_count,
    }
    write_json(workspace_root / "segment_metadata.json", segment_metadata)
    print(
        f"Prepared Naruto workspace with {extracted_frame_count} frames at {target_width}x{target_height} in {workspace_root}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
