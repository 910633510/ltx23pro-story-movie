#!/bin/bash
#SBATCH --job-name=hunyuan-smoke
#SBATCH --output=/home/%u/ltx23pro-story-movie/logs/hunyuan-smoke-%j.out
#SBATCH --error=/home/%u/ltx23pro-story-movie/logs/hunyuan-smoke-%j.err
#SBATCH --partition=general-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --constraint=a100,epyc64
#SBATCH --mem=192G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/ltx23pro-story-movie}"
CONFIG_JSON="${CONFIG_JSON:-$PROJECT_ROOT/config/hunyuan_i2v_smoke_test.json}"
MINICONDA_DIR="${MINICONDA_DIR:-$HOME/miniconda3}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-hunyuan15}"

mkdir -p "$PROJECT_ROOT/logs"
cd "$PROJECT_ROOT"

source "$MINICONDA_DIR/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV_NAME"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:128}"
export HF_HOME="${HF_HOME:-$PROJECT_ROOT/.hf_cache}"

hostname
nvidia-smi
python --version
python "$PROJECT_ROOT/scripts/run_hunyuan_i2v_smoke.py" --config "$CONFIG_JSON"
