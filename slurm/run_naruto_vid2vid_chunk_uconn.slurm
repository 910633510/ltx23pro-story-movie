#!/bin/bash
#SBATCH --job-name=naruto-v2v
#SBATCH --output=/home/%u/ltx23pro-story-movie/logs/naruto-v2v-%A_%a.out
#SBATCH --error=/home/%u/ltx23pro-story-movie/logs/naruto-v2v-%A_%a.err
#SBATCH --partition=general-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --constraint=a100,epyc64
#SBATCH --mem=192G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/ltx23pro-story-movie}"
CONFIG_JSON="${CONFIG_JSON:-$PROJECT_ROOT/config/naruto_379_vid2vid_4k_3min.json}"
CHUNK_MANIFEST="${CHUNK_MANIFEST:-$PROJECT_ROOT/refs/naruto_379/chunks_manifest.json}"
BOOTSTRAP_PYTHON="${BOOTSTRAP_PYTHON:-$HOME/miniconda3/envs/ltxbootstrap/bin/python}"

mkdir -p "$PROJECT_ROOT/logs"
cd "$PROJECT_ROOT"

"$BOOTSTRAP_PYTHON" "$PROJECT_ROOT/scripts/run_vid2vid_chunk.py" \
  --config "$CONFIG_JSON" \
  --manifest "$CHUNK_MANIFEST" \
  --chunk-index "$SLURM_ARRAY_TASK_ID"
