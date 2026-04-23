#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from naruto_vid2vid_lib import (
    chunk_windows,
    count_extracted_frames,
    ensure_dir,
    load_config,
    load_json,
    manifest_path_for_config,
    parse_frame_rate,
    require_ffmpeg,
    resolve_path,
    run,
    write_json,
)


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Create 4K Naruto vid2vid chunk videos and a chunk manifest.")
    parser.add_argument(
        "--config",
        type=Path,
        default=project_root / "config" / "naruto_379_vid2vid_4k_3min.json",
        help="Path to the Naruto vid2vid config JSON.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-encode chunk source videos even if they already exist.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    ffmpeg = require_ffmpeg()

    workspace_root = ensure_dir(resolve_path(config["workspace_root"], args.config.parent))
    source_frames_dir = workspace_root / "source_frames"
    if not source_frames_dir.exists():
        raise FileNotFoundError(f"Missing extracted frames: {source_frames_dir}")

    total_frames = count_extracted_frames(source_frames_dir)
    if total_frames <= 0:
        raise RuntimeError("No extracted source frames found. Run extract_naruto_segment.py first.")

    chunk_num_frames = int(config["chunk_num_frames"])
    overlap_frames = int(config["chunk_overlap_frames"])
    windows = chunk_windows(total_frames, chunk_num_frames, overlap_frames)

    _, fps_text = parse_frame_rate(config["frame_rate"])
    chunks_dir = ensure_dir(workspace_root / "chunks")
    pose_dir = ensure_dir(workspace_root / "control_pose")
    canny_dir = ensure_dir(workspace_root / "control_canny")
    output_root = ensure_dir(resolve_path(config["output_root"], args.config.parent))
    rendered_dir = ensure_dir(output_root / "rendered_chunks")
    manifest_path = manifest_path_for_config(config, args.config)
    existing_chunks: dict[str, dict[str, object]] = {}
    if manifest_path.exists():
        for existing in load_json(manifest_path).get("chunks", []):
            existing_chunks[str(existing["chunk_id"])] = existing

    manifest_chunks: list[dict[str, object]] = []
    for window in windows:
        chunk_id = f"chunk-{window['chunk_index']:04d}"
        source_video = chunks_dir / f"{chunk_id}_source.mp4"
        if args.force or not source_video.exists():
            run(
                [
                    ffmpeg,
                    "-y",
                    "-framerate",
                    fps_text,
                    "-start_number",
                    str(window["start_frame"]),
                    "-i",
                    str(source_frames_dir / "frame-%06d.png"),
                    "-frames:v",
                    str(window["num_frames"]),
                    "-c:v",
                    "libx264",
                    "-preset",
                    "medium",
                    "-crf",
                    "12",
                    "-pix_fmt",
                    "yuv420p",
                    str(source_video),
                ]
            )

        first_frame_path = source_frames_dir / f"frame-{window['start_frame']:06d}.png"
        existing = existing_chunks.get(chunk_id, {})
        manifest_chunks.append(
            {
                "chunk_id": chunk_id,
                "chunk_index": window["chunk_index"],
                "start_frame": window["start_frame"],
                "end_frame": window["end_frame"],
                "num_frames": window["num_frames"],
                "keep_start_offset": window["keep_start_offset"],
                "keep_end_offset": window["keep_end_offset"],
                "control_mode": existing.get("control_mode", "pending"),
                "source_video": str(source_video),
                "source_frames_dir": str(source_frames_dir),
                "first_frame_path": str(first_frame_path),
                "pose_control_video": str(pose_dir / f"{chunk_id}_pose.mp4"),
                "canny_control_video": str(canny_dir / f"{chunk_id}_canny.mp4"),
                "rendered_video": str(rendered_dir / f"{chunk_id}.mp4"),
                "pose_detected_frames": existing.get("pose_detected_frames"),
                "pose_total_frames": existing.get("pose_total_frames"),
                "pose_coverage": existing.get("pose_coverage"),
            }
        )

    manifest = {
        "project_name": config["project_name"],
        "config_path": str(args.config.resolve()),
        "source_video": str(resolve_path(config["source_video"], args.config.parent)),
        "source_frames_dir": str(source_frames_dir),
        "frame_rate": config["frame_rate"],
        "working_width": int(config["working_width"]),
        "working_height": int(config["working_height"]),
        "chunk_num_frames": chunk_num_frames,
        "chunk_overlap_frames": overlap_frames,
        "chunk_fallback_num_frames": int(config.get("chunk_fallback_num_frames", 65)),
        "total_source_frames": total_frames,
        "chunks": manifest_chunks,
    }
    write_json(manifest_path, manifest)
    print(f"Wrote {len(manifest_chunks)} chunk definitions to {manifest_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
