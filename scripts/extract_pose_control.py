#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from naruto_vid2vid_lib import load_config, load_json, manifest_path_for_config, write_json


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Extract MediaPipe pose skeleton control videos for Naruto vid2vid chunks.")
    parser.add_argument(
        "--config",
        type=Path,
        default=project_root / "config" / "naruto_379_vid2vid_4k_3min.json",
        help="Path to the Naruto vid2vid config JSON.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute pose control videos even if they already exist.",
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

    try:
        import mediapipe as mp
    except ImportError as exc:
        raise RuntimeError("mediapipe is required for pose extraction. Install it in the bootstrap env.") from exc

    pose_cfg = config["pose_control"]
    pose_threshold = float(config["pose_min_coverage"])

    mp_pose = mp.solutions.pose
    drawing_utils = mp.solutions.drawing_utils
    white_line = drawing_utils.DrawingSpec(color=(255, 255, 255), thickness=4, circle_radius=2)
    white_point = drawing_utils.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=3)

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=int(pose_cfg.get("model_complexity", 1)),
        enable_segmentation=False,
        min_detection_confidence=float(pose_cfg.get("detector_confidence", 0.35)),
        min_tracking_confidence=float(pose_cfg.get("tracking_confidence", 0.35)),
    ) as pose:
        for chunk in manifest["chunks"]:
            chunk_id = chunk["chunk_id"]
            source_video = Path(chunk["source_video"])
            pose_video = Path(chunk["pose_control_video"])

            if pose_video.exists() and not args.force:
                print(f"Keeping existing pose control: {pose_video.name}", flush=True)
                continue

            capture = cv2.VideoCapture(str(source_video))
            if not capture.isOpened():
                raise RuntimeError(f"Unable to open chunk source video: {source_video}")

            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
            writer = build_writer(pose_video, width, height, fps)

            detected_frames = 0
            total_frames = 0
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                total_frames += 1
                result = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                canvas = np.zeros_like(frame)
                if result.pose_landmarks:
                    detected_frames += 1
                    drawing_utils.draw_landmarks(
                        canvas,
                        result.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=white_point,
                        connection_drawing_spec=white_line,
                    )
                writer.write(canvas)

            capture.release()
            writer.release()

            coverage = detected_frames / total_frames if total_frames else 0.0
            chunk["pose_detected_frames"] = detected_frames
            chunk["pose_total_frames"] = total_frames
            chunk["pose_coverage"] = coverage
            chunk["control_mode"] = "pose" if coverage >= pose_threshold else "canny"
            print(
                f"{chunk_id}: pose coverage {coverage:.1%} -> {chunk['control_mode']}",
                flush=True,
            )

    write_json(manifest_path, manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
