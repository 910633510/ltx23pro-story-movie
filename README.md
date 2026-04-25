# LTX-2.3 Pro Story Movie

This project sets up a separate LTX-2.3 pipeline beside `wan-story-movie` so you can run higher-end story generation on UConn HPC with the official full LTX-2.3 checkpoint.

It uses the upstream [`Lightricks/LTX-2`](https://github.com/Lightricks/LTX-2) repo and the quality-first `ltx-2.3-22b-dev.safetensors` checkpoint, together with the current required stage-two assets:

- `ltx-2.3-22b-dev.safetensors`
- `ltx-2.3-22b-distilled-1.1.safetensors` for IC-LoRA video-to-video runs
- `ltx-2.3-spatial-upscaler-x2-1.1.safetensors`
- `ltx-2.3-22b-distilled-lora-384-1.1.safetensors`
- the gated Gemma text encoder repo `google/gemma-3-12b-it-qat-q4_0-unquantized`

## What This Project Does

- clones and installs the official `LTX-2` repository
- downloads the required LTX-2.3 assets
- submits a UConn HPC Slurm job
- generates scene-by-scene MP4 clips from a story JSON
- chains scenes by extracting the last frame of the previous clip and using it as image conditioning for the next clip
- optionally concatenates the clips into a single movie
- installs an `imageio-ffmpeg` fallback during bootstrap so HPC runs can create `movie.mp4` even when no system `ffmpeg` module is available
- includes a separate Naruto-379 native 4K vid2vid path built around the upstream `ltx_pipelines.ic_lora` pipeline
- includes a separate HunyuanVideo-1.5 image-to-video smoke test using the official `Tencent-Hunyuan/HunyuanVideo-1.5` source repo

## Important Limits

- LTX-2.3 is officially 4K-capable, but true 4K inference with the full 22B model is much heavier than the Wan pipeline.
- On UConn A100 40 GB nodes, the safest first run is not 4K. Start at `1920x1088` or `1536x1024`, verify the pipeline works, then try a higher-resolution profile.
- Do not use the `fp8-cast` quantization preset on A100 nodes. That path is meant for newer architectures and will fail on the A100s currently common in `general-gpu`.
- HunyuanVideo-1.5 is added here as a separate local open-source test path, not a replacement for the LTX story pipeline. The default smoke test uses 480p I2V step-distill plus 720p SR because that is the fastest meaningful quality check on a single A100.
- The Gemma repo is gated on Hugging Face. You must both:
  - accept the Gemma terms on Hugging Face
  - provide a token on HPC, for example `export HF_TOKEN=...`

## Project Layout

- `config/story.example.json`: starter story config
- `config/xianxia_fox_sword_5min_story.json`: long xianxia sample
- `config/xianxia_fox_sword_photoreal_5min_story.json`: long xianxia sample with stronger photoreal identity locking
- `config/xianxia_fox_sword_photoreal_ref_5min_story.json`: long xianxia sample that reapplies a reference image on every scene
- `config/xianxia_sword_fairy_photoreal_ref_5min_story.json`: reference-driven sword immortal + celestial fairy variant tuned for white/silver/blue dual-character images
- `config/xianxia_sword_fairy_photoreal_ref_4k_story.json`: premium continuous 4K override for the sword immortal + celestial fairy variant at `3840x2176`
- `config/naruto_379_vid2vid_4k_3min.json`: native 4K Naruto-379 vid2vid config for the first 180 seconds
- `config/hunyuan_i2v_smoke_test.json`: HunyuanVideo-1.5 I2V smoke-test config using the fairy reference image
- `refs/README.md`: where to place the dual-character reference image for the reference-driven run
- `refs/naruto_379/README.md`: generated workspace for the Naruto vid2vid pipeline
- `storyboards/xianxia_fox_sword_5min_script.md`: readable version of the same story
- `storyboards/xianxia_sword_fairy_5min_script.md`: readable sword immortal + celestial fairy version
- `scripts/download_ltx23_assets.py`: downloads the full model bundle
- `scripts/bootstrap_uconn_hpc.sh`: sets up env + models + optional Slurm submission
- `scripts/story_to_movie.py`: runs LTX scene-by-scene with progress logging
- `scripts/extract_naruto_segment.py`: extracts the first 180 seconds, audio, and frame sequence from `refs/Naruto-379.mp4`
- `scripts/make_vid2vid_chunks.py`: creates 97-frame chunk videos and the chunk manifest
- `scripts/extract_pose_control.py`: builds pose skeleton control videos and marks pose-vs-canny chunk routing
- `scripts/extract_canny_control.py`: builds canny edge control videos for every chunk
- `scripts/run_vid2vid_chunk.py`: renders one chunk through `ltx_pipelines.ic_lora`
- `scripts/assemble_vid2vid_output.py`: trims overlap and assembles `final_silent.mp4`
- `scripts/mux_original_audio.py`: muxes the original Japanese audio back onto the assembled result
- `scripts/run_vid2vid_slurm_array.sh`: prepares the Naruto workspace and submits the Slurm array
- `scripts/bootstrap_hunyuan_hpc.sh`: clones official HunyuanVideo-1.5, creates the conda env, and downloads smoke-test models
- `scripts/run_hunyuan_i2v_smoke.py`: validates config and calls official Hunyuan `generate.py`
- `slurm/run_story_movie_uconn.slurm`: UConn Storrs batch job
- `slurm/run_naruto_vid2vid_chunk_uconn.slurm`: UConn Storrs Slurm array worker for Naruto vid2vid chunks
- `slurm/run_hunyuan_i2v_smoke_uconn.slurm`: UConn Storrs single-A100 Hunyuan smoke test

## HPC Setup

Copy the project to HPC from your Mac first:

```bash
scp -r /Users/xue/Project/ltx23pro-story-movie xiw20029@hpc2.storrs.hpc.uconn.edu:~
```

Then SSH into UConn HPC and run:

```bash
cd ~/ltx23pro-story-movie
HF_TOKEN=YOUR_HF_TOKEN ./scripts/bootstrap_uconn_hpc.sh --submit
```

If your group requires a PI account:

```bash
cd ~/ltx23pro-story-movie
PI_ACCOUNT=YOUR_PI_ACCOUNT HF_TOKEN=YOUR_HF_TOKEN ./scripts/bootstrap_uconn_hpc.sh --submit
```

For direct Slurm submission with a PI account, pass the account at submit time instead of editing the Slurm file:

```bash
STORY_JSON="$HOME/ltx23pro-story-movie/config/xianxia_sword_fairy_photoreal_ref_5min_story.json" sbatch --account=YOUR_PI_ACCOUNT slurm/run_story_movie_uconn.slurm
```

The bootstrap script will:

1. install Miniconda if needed
2. create a small bootstrap env with Python 3.10
3. clone/update the official `LTX-2` repo under `vendor/LTX-2`
4. run `uv sync --frozen`
5. download the LTX-2.3 full-model assets plus Gemma
6. optionally submit the Slurm job

For the Naruto vid2vid pipeline you also need the distilled checkpoint and the official pose/canny IC-LoRAs. Bootstrap them with:

```bash
cd ~/ltx23pro-story-movie
DOWNLOAD_DISTILLED_CHECKPOINT=1 DOWNLOAD_NARUTO_IC_LORAS=1 HF_TOKEN=YOUR_HF_TOKEN ./scripts/bootstrap_uconn_hpc.sh
```

## Monitoring

After submit:

```bash
squeue -u xiw20029
tail -f "$(ls -t ~/ltx23pro-story-movie/logs/ltx-story-*.out | head -n 1)"
tail -f "$(ls -t ~/ltx23pro-story-movie/logs/ltx-story-*.err | head -n 1)"
```

Check outputs:

```bash
ls -lh ~/ltx23pro-story-movie/outputs/<jobid>
find ~/ltx23pro-story-movie/outputs/<jobid> -maxdepth 1 -type f | sort
```

## Story Config Shape

`story_to_movie.py` reads JSON in this format:

```json
{
  "movie_title": "Example",
  "default_width": 1920,
  "default_height": 1088,
  "default_num_frames": 121,
  "default_frame_rate": 24,
  "default_num_inference_steps": 30,
  "default_seed": 5000,
  "pipeline_module": "ltx_pipelines.ti2vid_two_stages",
  "quantization": null,
  "streaming_prefetch_count": 2,
  "max_batch_size": 1,
  "enhance_prompt": true,
  "prompt_prefix": "optional global style and identity lock text",
  "prompt_suffix": "optional global style reminder",
  "reference_image": "~/ltx23pro-story-movie/refs/xianxia_duo_reference.png",
  "reference_image_mode": "first_scene",
  "reference_image_strength": 0.92,
  "chain_strength": 0.75,
  "negative_prompt": "optional negative prompt",
  "scenes": [
    {
      "id": "scene-01",
      "prompt": "Detailed cinematic prompt here",
      "continue_from_previous": false
    }
  ]
}
```

Useful per-scene overrides:

- `seed`
- `width`
- `height`
- `num_frames`
- `num_inference_steps`
- `frame_rate`
- `continue_from_previous`
- `chain_strength`
- `prompt_prefix`
- `prompt_suffix`
- `reference_image`
- `reference_image_mode`: `always`, `first_scene`, `when_no_chain`, or `none`
- `reference_image_strength`
- `start_image`
- `start_image_strength`
- `enhance_prompt`

Story files can also use `"extends": "other_story.json"` to inherit a base story and override only selected top-level settings such as width, height, prompt prefix, chain strength, or seed.

## Reference-Driven Runs

If you care more about stable live-action faces than pure prompt freedom, use a reference-driven config.

1. Put a dual-character reference image at:

```bash
~/ltx23pro-story-movie/refs/xianxia_duo_reference.png
```

2. Submit the reference-driven xianxia story:

```bash
cd ~/ltx23pro-story-movie
STORY_JSON="$HOME/ltx23pro-story-movie/config/xianxia_fox_sword_photoreal_ref_5min_story.json" sbatch slurm/run_story_movie_uconn.slurm
```

This keeps the usual previous-frame chaining, but also reapplies the same character reference image every scene.

For continuity-heavy runs, prefer `reference_image_mode: "first_scene"` so the reference image seeds only the first clip and later scenes chain from the previous clip's last frame. Using `always` can make every scene restart from the same reference image.

For a sword-immortal plus celestial-fairy run, place your image at:

```bash
~/ltx23pro-story-movie/refs/xianxia_fairy_duo_reference.png
```

Then submit:

```bash
cd ~/ltx23pro-story-movie
STORY_JSON="$HOME/ltx23pro-story-movie/config/xianxia_sword_fairy_photoreal_ref_5min_story.json" sbatch slurm/run_story_movie_uconn.slurm
```

For the 4K version:

```bash
cd ~/ltx23pro-story-movie
STORY_JSON="$HOME/ltx23pro-story-movie/config/xianxia_sword_fairy_photoreal_ref_4k_story.json" sbatch slurm/run_story_movie_uconn.slurm
```

The 4K fairy config keeps `24` inference steps to stay closer to the 12-hour `general-gpu` wall-time limit, uses `3840x2176`, lowers reference image strength so the first clip is not just a static portrait, and raises chaining strength to improve shot-to-shot continuity.

## Safer First Runs

Recommended first submit:

- partition: `general-gpu`
- constraint: `a100,epyc64`
- GPU request: `--gres=gpu:1`
- width: `1920`
- height: `1088`
- frames: `121`
- steps: `20` to `30`
- quantization: `null` on A100 nodes

If that works, then try heavier settings. If you later run on Hopper-class GPUs, you can experiment with `fp8-cast`. For a true 4K attempt, set:

- width: `3840`
- height: `2176`

But expect much higher memory pressure and runtime.

## HunyuanVideo-1.5 I2V Smoke Test

This path is separate from `story_to_movie.py`. It uses the official [`Tencent-Hunyuan/HunyuanVideo-1.5`](https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5) repo and the Hugging Face model [`tencent/HunyuanVideo-1.5`](https://huggingface.co/tencent/HunyuanVideo-1.5).

The default test intentionally uses the official `480p_i2v_step_distilled` model with 12 steps and enables the official 480p-to-720p SR stage. It is meant to answer "is Hunyuan visually better for our reference-image live-action test?" before spending A100 hours on heavier settings.

One-time setup on UConn HPC:

```bash
cd ~/ltx23pro-story-movie
git pull --ff-only origin main
HF_TOKEN=YOUR_HF_TOKEN ./scripts/bootstrap_hunyuan_hpc.sh
```

If your group requires a PI account and you want setup plus immediate submit:

```bash
cd ~/ltx23pro-story-movie
PI_ACCOUNT=YOUR_PI_ACCOUNT HF_TOKEN=YOUR_HF_TOKEN ./scripts/bootstrap_hunyuan_hpc.sh --submit
```

The `HF_TOKEN` must have access to `black-forest-labs/FLUX.1-Redux-dev`, because the official Hunyuan I2V pipeline uses that repo as its SigLIP vision encoder. Accept access on Hugging Face before running bootstrap.

Submit only the smoke test:

```bash
cd ~/ltx23pro-story-movie
CONFIG_JSON="$HOME/ltx23pro-story-movie/config/hunyuan_i2v_smoke_test.json" sbatch slurm/run_hunyuan_i2v_smoke_uconn.slurm
```

Monitor it:

```bash
squeue -u "$USER"
tail -f "$(ls -t ~/ltx23pro-story-movie/logs/hunyuan-smoke-*.err | head -n 1)"
```

Expected outputs:

- `outputs/hunyuan_i2v_smoke/hunyuan_i2v_smoke.mp4`
- `outputs/hunyuan_i2v_smoke/hunyuan_i2v_smoke_before_sr.mp4`
- `outputs/hunyuan_i2v_smoke/hunyuan_i2v_smoke_config.json`
- `outputs/hunyuan_i2v_smoke/hunyuan_i2v_smoke_command.sh`

## Naruto-379 Native 4K Vid2Vid

This path is separate from `story_to_movie.py`. It uses the upstream `ltx_pipelines.ic_lora` CLI so the source anime video becomes a chunked control signal rather than a prompt-only storyboard.

Source asset expected in repo:

```bash
~/ltx23pro-story-movie/refs/Naruto-379.mp4
```

The config is:

```bash
~/ltx23pro-story-movie/config/naruto_379_vid2vid_4k_3min.json
```

What the Naruto config locks today:

- input window: `0s-180s`
- resolution: `3840x2160`
- fps: `24000/1001`
- chunk size: `97` frames
- overlap: `16` frames
- preferred control: MediaPipe pose skeletons
- fallback control: canny edges
- output root: `outputs/naruto_379_vid2vid_4k_3min`

Recommended HPC flow:

```bash
cd ~/ltx23pro-story-movie
git pull --ff-only origin main
DOWNLOAD_DISTILLED_CHECKPOINT=1 DOWNLOAD_NARUTO_IC_LORAS=1 HF_TOKEN=YOUR_HF_TOKEN ./scripts/bootstrap_uconn_hpc.sh
./scripts/run_vid2vid_slurm_array.sh --config "$HOME/ltx23pro-story-movie/config/naruto_379_vid2vid_4k_3min.json"
```

If your group requires a PI account:

```bash
./scripts/run_vid2vid_slurm_array.sh --account YOUR_PI_ACCOUNT --config "$HOME/ltx23pro-story-movie/config/naruto_379_vid2vid_4k_3min.json"
```

That wrapper does four things before submission:

1. extracts the first 180 seconds into `refs/naruto_379/source_segment.mp4`, `source_audio.m4a`, and `source_frames/`
2. creates 97-frame source chunks and `refs/naruto_379/chunks_manifest.json`
3. builds pose skeleton controls and marks chunks that have enough pose coverage
4. builds canny fallback controls, then submits a `%1` Slurm array over all chunks

Array output clips land in:

```bash
~/ltx23pro-story-movie/outputs/naruto_379_vid2vid_4k_3min/rendered_chunks
```

After the array finishes:

```bash
cd ~/ltx23pro-story-movie
$HOME/miniconda3/envs/ltxbootstrap/bin/python scripts/assemble_vid2vid_output.py --config "$HOME/ltx23pro-story-movie/config/naruto_379_vid2vid_4k_3min.json"
$HOME/miniconda3/envs/ltxbootstrap/bin/python scripts/mux_original_audio.py --config "$HOME/ltx23pro-story-movie/config/naruto_379_vid2vid_4k_3min.json"
```

Outputs:

- `outputs/naruto_379_vid2vid_4k_3min/final_silent.mp4`
- `outputs/naruto_379_vid2vid_4k_3min/final_with_audio.mp4`

Notes:

- The config keeps a `negative_prompt` field for documentation, but the current upstream distilled `ic_lora` CLI does not consume negative prompts.
- The pipeline defaults to the official pose IC-LoRA when MediaPipe pose coverage reaches `70%` for a chunk; otherwise that chunk routes to the official canny IC-LoRA.
- The config reserves `chunk_fallback_num_frames=65` for manual OOM recovery, but the default manifest stays on `97`-frame chunks for cleaner overlap assembly.

## Local Notes

This Mac can scaffold and syntax-check the pipeline wrapper, but it cannot run real LTX inference because the model expects an NVIDIA CUDA GPU.
