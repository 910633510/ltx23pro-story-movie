#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from naruto_vid2vid_lib import ensure_dir, load_config, require_ffmpeg, resolve_path, run


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Mux the original Naruto audio back onto final_silent.mp4.")
    parser.add_argument(
        "--config",
        type=Path,
        default=project_root / "config" / "naruto_379_vid2vid_4k_3min.json",
        help="Path to the Naruto vid2vid config JSON.",
    )
    parser.add_argument(
        "--silent-video",
        type=Path,
        default=None,
        help="Path to the assembled silent MP4. Defaults to outputs/.../final_silent.mp4",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Path for the muxed MP4. Defaults to outputs/.../final_with_audio.mp4",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    ffmpeg = require_ffmpeg()

    workspace_root = resolve_path(config["workspace_root"], args.config.parent)
    output_root = ensure_dir(resolve_path(config["output_root"], args.config.parent))
    silent_video = args.silent_video or (output_root / "final_silent.mp4")
    audio_path = workspace_root / "source_audio.m4a"
    output_path = args.output_path or (output_root / "final_with_audio.mp4")

    if not silent_video.exists():
        raise FileNotFoundError(f"Missing silent assembled video: {silent_video}")
    if not audio_path.exists():
        raise FileNotFoundError(f"Missing extracted source audio: {audio_path}")

    run(
        [
            ffmpeg,
            "-y",
            "-i",
            str(silent_video),
            "-i",
            str(audio_path),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "copy",
            "-c:a",
            "copy",
            "-shortest",
            str(output_path),
        ]
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
