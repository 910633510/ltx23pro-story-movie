#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CONFIG_JSON="${CONFIG_JSON:-$PROJECT_ROOT/config/naruto_379_vid2vid_4k_3min.json}"
SLURM_FILE="${SLURM_FILE:-$PROJECT_ROOT/slurm/run_naruto_vid2vid_chunk_uconn.slurm}"
BOOTSTRAP_PYTHON="${BOOTSTRAP_PYTHON:-$HOME/miniconda3/envs/ltxbootstrap/bin/python}"
PI_ACCOUNT="${PI_ACCOUNT:-}"
SKIP_PREPARE=false
FORCE_PREPARE=false
ARRAY_PARALLELISM=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG_JSON="$2"
      shift 2
      ;;
    --account)
      PI_ACCOUNT="$2"
      shift 2
      ;;
    --skip-prepare)
      SKIP_PREPARE=true
      shift
      ;;
    --force-prepare)
      FORCE_PREPARE=true
      shift
      ;;
    --array-parallelism)
      ARRAY_PARALLELISM="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

if [[ ! -x "$BOOTSTRAP_PYTHON" ]]; then
  echo "Bootstrap Python not found: $BOOTSTRAP_PYTHON" >&2
  exit 1
fi

prepare_args=()
if $FORCE_PREPARE; then
  prepare_args+=(--force)
fi

if ! $SKIP_PREPARE; then
  "$BOOTSTRAP_PYTHON" "$PROJECT_ROOT/scripts/extract_naruto_segment.py" --config "$CONFIG_JSON" "${prepare_args[@]}"
  "$BOOTSTRAP_PYTHON" "$PROJECT_ROOT/scripts/make_vid2vid_chunks.py" --config "$CONFIG_JSON" "${prepare_args[@]}"
  "$BOOTSTRAP_PYTHON" "$PROJECT_ROOT/scripts/extract_pose_control.py" --config "$CONFIG_JSON" "${prepare_args[@]}"
  "$BOOTSTRAP_PYTHON" "$PROJECT_ROOT/scripts/extract_canny_control.py" --config "$CONFIG_JSON" "${prepare_args[@]}"
fi

CHUNK_MANIFEST="$("$BOOTSTRAP_PYTHON" - "$CONFIG_JSON" <<'PY'
import json
import sys
from pathlib import Path

config_path = Path(sys.argv[1]).resolve()
config = json.loads(config_path.read_text(encoding="utf-8"))
manifest = config.get("chunk_manifest_path")
if manifest is None:
    manifest_path = (config_path.parent / config["workspace_root"] / "chunks_manifest.json").resolve()
else:
    manifest_path = (Path(manifest) if Path(manifest).is_absolute() else (config_path.parent / manifest)).resolve()
print(manifest_path)
PY
)"

CHUNK_COUNT="$("$BOOTSTRAP_PYTHON" - "$CHUNK_MANIFEST" <<'PY'
import json
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(len(manifest["chunks"]))
PY
)"

if [[ "$CHUNK_COUNT" -le 0 ]]; then
  echo "Chunk manifest is empty: $CHUNK_MANIFEST" >&2
  exit 1
fi

if [[ -z "$ARRAY_PARALLELISM" ]]; then
  ARRAY_PARALLELISM="$("$BOOTSTRAP_PYTHON" - "$CONFIG_JSON" <<'PY'
import json
import sys
from pathlib import Path

config = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(int(config.get("slurm", {}).get("array_parallelism", 1)))
PY
)"
fi

sbatch_args=(--array "0-$((CHUNK_COUNT - 1))%$ARRAY_PARALLELISM")
if [[ -n "$PI_ACCOUNT" ]]; then
  sbatch_args+=(--account "$PI_ACCOUNT")
fi

CONFIG_JSON="$CONFIG_JSON" CHUNK_MANIFEST="$CHUNK_MANIFEST" sbatch "${sbatch_args[@]}" "$SLURM_FILE"
