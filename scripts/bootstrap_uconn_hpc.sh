#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
LTX_REPO_DIR="${LTX_REPO_DIR:-$PROJECT_ROOT/vendor/LTX-2}"
LTX_PYTHON="${LTX_PYTHON:-$LTX_REPO_DIR/.venv/bin/python}"
SLURM_FILE="${SLURM_FILE:-$PROJECT_ROOT/slurm/run_story_movie_uconn.slurm}"
MINICONDA_DIR="${MINICONDA_DIR:-$HOME/miniconda3}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-ltxbootstrap}"
PI_ACCOUNT="${PI_ACCOUNT:-}"
STORY_JSON="${STORY_JSON:-$PROJECT_ROOT/config/story.example.json}"
SUBMIT_JOB=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --submit)
      SUBMIT_JOB=true
      shift
      ;;
    --account)
      PI_ACCOUNT="$2"
      shift 2
      ;;
    --env-name)
      CONDA_ENV_NAME="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

mkdir -p "$PROJECT_ROOT/vendor" "$PROJECT_ROOT/models" "$PROJECT_ROOT/logs" "$PROJECT_ROOT/outputs"

if [[ ! -f "$MINICONDA_DIR/etc/profile.d/conda.sh" ]]; then
  INSTALLER="/tmp/Miniconda3-latest-Linux-x86_64.sh"
  if command -v curl >/dev/null 2>&1; then
    curl -L -o "$INSTALLER" https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
  elif command -v wget >/dev/null 2>&1; then
    wget -O "$INSTALLER" https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
  else
    echo "Need curl or wget to install Miniconda." >&2
    exit 1
  fi
  bash "$INSTALLER" -b -p "$MINICONDA_DIR"
fi

source "$MINICONDA_DIR/etc/profile.d/conda.sh"
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main >/dev/null 2>&1 || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r >/dev/null 2>&1 || true

if ! conda env list | awk '{print $1}' | grep -qx "$CONDA_ENV_NAME"; then
  conda create -n "$CONDA_ENV_NAME" python=3.10 -y
fi

conda activate "$CONDA_ENV_NAME"
python -m pip install --upgrade pip setuptools wheel
python -m pip install uv "huggingface_hub[cli]" opencv-python mediapipe imageio-ffmpeg

if [[ ! -d "$LTX_REPO_DIR/.git" ]]; then
  git clone https://github.com/Lightricks/LTX-2.git "$LTX_REPO_DIR"
else
  git -C "$LTX_REPO_DIR" pull --ff-only
fi

(
  cd "$LTX_REPO_DIR"
  uv sync --frozen
)

uv pip install --python "$LTX_PYTHON" imageio-ffmpeg

download_args=()
if [[ "${DOWNLOAD_DISTILLED_CHECKPOINT:-0}" == "1" ]]; then
  download_args+=(--include-distilled-checkpoint)
fi
if [[ "${DOWNLOAD_NARUTO_IC_LORAS:-0}" == "1" ]]; then
  download_args+=(--include-naruto-ic-loras)
fi

python "$PROJECT_ROOT/scripts/download_ltx23_assets.py" "${download_args[@]}"

if [[ ! -d "$PROJECT_ROOT/models/gemma-3-12b-it-qat-q4_0-unquantized" ]] || [[ -z "$(find "$PROJECT_ROOT/models/gemma-3-12b-it-qat-q4_0-unquantized" -maxdepth 1 -type f 2>/dev/null)" ]]; then
  cat >&2 <<'EOF'
Gemma assets are missing. The Gemma repo is gated on Hugging Face.
Accept the Gemma terms, export HF_TOKEN on HPC, and rerun bootstrap.
EOF
  exit 1
fi

if $SUBMIT_JOB; then
  sbatch_args=()
  if [[ -n "$PI_ACCOUNT" ]]; then
    sbatch_args+=(--account "$PI_ACCOUNT")
  fi
  STORY_JSON="$STORY_JSON" sbatch "${sbatch_args[@]}" "$SLURM_FILE"
fi

echo "Bootstrap complete."
