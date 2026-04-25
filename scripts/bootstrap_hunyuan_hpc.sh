#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
HUNYUAN_REPO_DIR="${HUNYUAN_REPO_DIR:-$PROJECT_ROOT/vendor/HunyuanVideo-1.5}"
HUNYUAN_MODEL_DIR="${HUNYUAN_MODEL_DIR:-$PROJECT_ROOT/models/HunyuanVideo-1.5}"
MINICONDA_DIR="${MINICONDA_DIR:-$HOME/miniconda3}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-hunyuan15}"
CONFIG_JSON="${CONFIG_JSON:-$PROJECT_ROOT/config/hunyuan_i2v_smoke_test.json}"
SLURM_FILE="${SLURM_FILE:-$PROJECT_ROOT/slurm/run_hunyuan_i2v_smoke_uconn.slurm}"
PI_ACCOUNT="${PI_ACCOUNT:-}"
SUBMIT_JOB=false
FULL_DOWNLOAD=false
SKIP_DOWNLOAD=false

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
    --full-download)
      FULL_DOWNLOAD=true
      shift
      ;;
    --skip-download)
      SKIP_DOWNLOAD=true
      shift
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

mkdir -p "$PROJECT_ROOT/vendor" "$PROJECT_ROOT/models" "$PROJECT_ROOT/logs" "$PROJECT_ROOT/outputs" "$HUNYUAN_MODEL_DIR"

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
python -m pip install "huggingface_hub[cli]" modelscope imageio-ffmpeg

if [[ ! -d "$HUNYUAN_REPO_DIR/.git" ]]; then
  git clone https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5.git "$HUNYUAN_REPO_DIR"
else
  git -C "$HUNYUAN_REPO_DIR" pull --ff-only
fi

python -m pip install -r "$HUNYUAN_REPO_DIR/requirements.txt"
python -m pip install --upgrade tencentcloud-sdk-python

if ! $SKIP_DOWNLOAD; then
  if $FULL_DOWNLOAD; then
    hf download tencent/HunyuanVideo-1.5 --local-dir "$HUNYUAN_MODEL_DIR"
  else
    hf download tencent/HunyuanVideo-1.5 \
      --local-dir "$HUNYUAN_MODEL_DIR" \
      --include "config.json" \
      --include "scheduler/*" \
      --include "vae/*" \
      --include "transformer/480p_i2v_step_distilled/*" \
      --include "transformer/720p_sr_distilled/*" \
      --include "upsampler/720p_sr_distilled/*"
  fi

  hf download Qwen/Qwen2.5-VL-7B-Instruct --local-dir "$HUNYUAN_MODEL_DIR/text_encoder/llm"
  hf download google/byt5-small --local-dir "$HUNYUAN_MODEL_DIR/text_encoder/byt5-small"
  modelscope download --model AI-ModelScope/Glyph-SDXL-v2 --local_dir "$HUNYUAN_MODEL_DIR/text_encoder/Glyph-SDXL-v2"

  flux_token_args=()
  if [[ -n "${HF_TOKEN:-}" ]]; then
    flux_token_args=(--token "$HF_TOKEN")
  elif [[ -n "${HUGGINGFACE_HUB_TOKEN:-}" ]]; then
    flux_token_args=(--token "$HUGGINGFACE_HUB_TOKEN")
  fi

  if ! hf download black-forest-labs/FLUX.1-Redux-dev \
    --local-dir "$HUNYUAN_MODEL_DIR/vision_encoder/siglip" \
    "${flux_token_args[@]}"; then
    cat >&2 <<'EOF'
Failed to download FLUX.1-Redux-dev, which HunyuanVideo-1.5 uses as its SigLIP vision encoder.
Accept access on Hugging Face, export HF_TOKEN on HPC, then rerun this bootstrap.
EOF
    exit 1
  fi
fi

python "$PROJECT_ROOT/scripts/run_hunyuan_i2v_smoke.py" --config "$CONFIG_JSON" --dry-run

if $SUBMIT_JOB; then
  sbatch_args=()
  if [[ -n "$PI_ACCOUNT" ]]; then
    sbatch_args+=(--account "$PI_ACCOUNT")
  fi
  CONFIG_JSON="$CONFIG_JSON" sbatch "${sbatch_args[@]}" "$SLURM_FILE"
fi

echo "Hunyuan bootstrap complete."
