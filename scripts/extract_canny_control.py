#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from naruto_vid2vid_lib import load_config, load_json, manifest_path_for_config, write_json


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Extract Canny edge control videos for Naruto vid2vid chunks.")
    parser.add_argument(
        "--config",
        type=Path,
        default=project_root / "config" / "naruto_379_vid2vid_4k_3min.json",
        help="Path to the Naruto vid2vid config JSON.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute canny control videos even if they already exist.",
    )
    return parser.parse_args()


def build_writer(path: Path, width: int, height: int, fps: float) -> cv2.VideoWriter:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Unable to create video writer for {path}")
    return writer


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    manifest_path = manifest_path_for_config(config, args.config)
    manifest = load_json(manifest_path)
    canny_cfg = config["canny_control"]

    blur_kernel = int(canny_cfg.get("blur_kernel", 5))
    if blur_kernel % 2 == 0:
        blur_kernel += 1
    low_threshold = int(canny_cfg.get("low_threshold", 100))
    high_threshold = int(canny_cfg.get("high_threshold", 200))

    for chunk in manifest["chunks"]:
        source_video = Path(chunk["source_video"])
        canny_video = Path(chunk["canny_control_video"])

        if canny_video.exists() and not args.force:
            print(f"Keeping existing canny control: {canny_video.name}", flush=True)
            continue

        capture = cv2.VideoCapture(str(source_video))
        if not capture.isOpened():
            raise RuntimeError(f"Unable to open chunk source video: {source_video}")

        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        writer = build_writer(canny_video, width, height, fps)

        while True:
            ok, frame = capture.read()
            if not ok:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)
            edges = cv2.Canny(blurred, low_threshold, high_threshold)
            writer.write(cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR))

        capture.release()
        writer.release()

    write_json(manifest_path, manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
