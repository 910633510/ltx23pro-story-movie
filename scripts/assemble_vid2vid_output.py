#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from naruto_vid2vid_lib import load_config, load_json, manifest_path_for_config, parse_frame_rate


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Assemble rendered Naruto vid2vid chunks into final_silent.mp4.")
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
        "--output-path",
        type=Path,
        default=None,
        help="Output path for the silent assembled MP4.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    manifest_path = args.manifest or manifest_path_for_config(config, args.config)
    manifest = load_json(manifest_path)

    try:
        import imageio_ffmpeg
    except ImportError as exc:
        raise RuntimeError("imageio-ffmpeg is required for chunk assembly.") from exc

    fps, _ = parse_frame_rate(config["frame_rate"])
    width = int(config["working_width"])
    height = int(config["working_height"])
    output_root = Path(config["output_root"])
    if not output_root.is_absolute():
        output_root = (args.config.parent / output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    output_path = args.output_path or (output_root / "final_silent.mp4")

    writer = imageio_ffmpeg.write_frames(
        str(output_path),
        (width, height),
        fps=fps,
        codec="libx264",
        macro_block_size=None,
        output_params=["-pix_fmt", "yuv420p", "-crf", "18", "-preset", "medium"],
    )
    writer.send(None)

    total_written_frames = 0
    for chunk in manifest["chunks"]:
        rendered_video = Path(chunk["rendered_video"])
        if not rendered_video.exists():
            raise FileNotFoundError(f"Missing rendered chunk: {rendered_video}")

        keep_start = int(chunk["keep_start_offset"])
        keep_end = int(chunk["keep_end_offset"])
        capture = cv2.VideoCapture(str(rendered_video))
        if not capture.isOpened():
            raise RuntimeError(f"Unable to open rendered chunk: {rendered_video}")

        frame_index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if keep_start <= frame_index <= keep_end:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                writer.send(rgb.tobytes())
                total_written_frames += 1
            frame_index += 1
        capture.release()

    writer.close()
    print(f"Assembled {total_written_frames} frames into {output_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
