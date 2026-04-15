#!/bin/bash
#SBATCH --job-name=ltx-story
#SBATCH --output=/home/%u/ltx23pro-story-movie/logs/ltx-story-%j.out
#SBATCH --error=/home/%u/ltx23pro-story-movie/logs/ltx-story-%j.err
#SBATCH --partition=general-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --constraint=epyc64
#SBATCH --mem=192G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --account=PINetidHere

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/ltx23pro-story-movie}"
LTX_REPO_DIR="$PROJECT_ROOT/vendor/LTX-2"
LTX_PYTHON="$LTX_REPO_DIR/.venv/bin/python"
STORY_JSON="${STORY_JSON:-$PROJECT_ROOT/config/story.example.json}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/outputs/$SLURM_JOB_ID}"

mkdir -p "$OUTPUT_DIR" "$PROJECT_ROOT/logs"

cd "$PROJECT_ROOT"

"$LTX_PYTHON" "$PROJECT_ROOT/scripts/story_to_movie.py" \
  --story "$STORY_JSON" \
  --ltx-repo "$LTX_REPO_DIR" \
  --ltx-python "$LTX_PYTHON" \
  --output-dir "$OUTPUT_DIR" \
  --concat

