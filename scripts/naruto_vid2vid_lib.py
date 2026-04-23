#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
from fractions import Fraction
from pathlib import Path
from typing import Any


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_path(value: str | os.PathLike[str], base_dir: Path | None = None) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (base_dir or project_root()) / path
    return path.resolve()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_config(path: Path) -> dict[str, Any]:
    data = load_json(path)
    if "extends" in data:
        base_path = resolve_path(data["extends"], path.parent)
        base_data = load_config(base_path)
        override = {key: value for key, value in data.items() if key != "extends"}
        data = {**base_data, **override}
    return data


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def parse_frame_rate(value: str | int | float) -> tuple[float, str]:
    if isinstance(value, (int, float)):
        fps = float(value)
        return fps, f"{fps:.12g}"
    text = str(value).strip()
    if "/" in text:
        fps = float(Fraction(text))
        return fps, text
    fps = float(text)
    return fps, text


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def manifest_path_for_config(config: dict[str, Any], config_path: Path) -> Path:
    if "chunk_manifest_path" in config:
        return resolve_path(config["chunk_manifest_path"], config_path.parent)
    workspace_root = resolve_path(config["workspace_root"], config_path.parent)
    return workspace_root / "chunks_manifest.json"


def count_extracted_frames(source_frames_dir: Path) -> int:
    return sum(1 for _ in source_frames_dir.glob("frame-*.png"))


def probe_video(path: Path) -> dict[str, Any]:
    import cv2

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video: {path}")

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    capture.release()

    if width <= 0 or height <= 0 or frame_count <= 0:
        raise RuntimeError(f"Failed to probe video metadata from {path}")

    duration_seconds = frame_count / fps if fps > 0 else 0.0
    return {
        "width": width,
        "height": height,
        "fps": fps,
        "frame_count": frame_count,
        "duration_seconds": duration_seconds,
    }


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


def require_ffmpeg() -> str:
    ffmpeg = find_ffmpeg()
    if ffmpeg is None:
        raise FileNotFoundError("ffmpeg not found. Install imageio-ffmpeg or set FFMPEG_BIN.")
    return ffmpeg


def run(command: list[str], cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    printable = " ".join(shlex.quote(part) for part in command)
    print(f"+ {printable}", flush=True)
    subprocess.run(command, cwd=cwd, env=env, check=True)


def chunk_windows(total_frames: int, chunk_num_frames: int, overlap_frames: int) -> list[dict[str, int]]:
    if total_frames <= 0:
        raise ValueError("total_frames must be positive")
    if chunk_num_frames <= 0:
        raise ValueError("chunk_num_frames must be positive")
    if overlap_frames < 0:
        raise ValueError("overlap_frames must be non-negative")
    if overlap_frames >= chunk_num_frames:
        raise ValueError("overlap_frames must be smaller than chunk_num_frames")
    if (chunk_num_frames - 1) % 8 != 0:
        raise ValueError("chunk_num_frames must satisfy 8k+1 for LTX-2 pipelines")

    stride = chunk_num_frames - overlap_frames
    starts = [1]
    if total_frames > chunk_num_frames:
        while True:
            next_start = starts[-1] + stride
            if next_start + chunk_num_frames - 1 >= total_frames:
                final_start = max(1, total_frames - chunk_num_frames + 1)
                if final_start > starts[-1]:
                    starts.append(final_start)
                break
            starts.append(next_start)

    windows: list[dict[str, int]] = []
    for index, start_frame in enumerate(starts, start=1):
        end_frame = min(total_frames, start_frame + chunk_num_frames - 1)
        num_frames = end_frame - start_frame + 1
        if (num_frames - 1) % 8 != 0:
            raise ValueError(
                f"Chunk {index} has {num_frames} frames; every chunk must satisfy 8k+1."
            )
        windows.append(
            {
                "chunk_index": index,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "num_frames": num_frames,
            }
        )

    for index, window in enumerate(windows):
        prev_overlap = 0
        next_overlap = 0
        if index > 0:
            prev = windows[index - 1]
            prev_overlap = max(0, prev["end_frame"] - window["start_frame"] + 1)
        if index + 1 < len(windows):
            next_window = windows[index + 1]
            next_overlap = max(0, window["end_frame"] - next_window["start_frame"] + 1)

        keep_start_offset = (prev_overlap + 1) // 2 if index > 0 else 0
        keep_end_offset = (window["num_frames"] - 1) - (next_overlap // 2)

        if keep_end_offset < keep_start_offset:
            raise ValueError(f"Invalid keep range for chunk {window['chunk_index']}")

        window["keep_start_offset"] = keep_start_offset
        window["keep_end_offset"] = keep_end_offset

    return windows
