# LTX-2.3 Pro Story Movie

This project sets up a separate LTX-2.3 pipeline beside `wan-story-movie` so you can run higher-end story generation on UConn HPC with the official full LTX-2.3 checkpoint.

It uses the upstream [`Lightricks/LTX-2`](https://github.com/Lightricks/LTX-2) repo and the quality-first `ltx-2.3-22b-dev.safetensors` checkpoint, together with the current required stage-two assets:

- `ltx-2.3-22b-dev.safetensors`
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

## Important Limits

- LTX-2.3 is officially 4K-capable, but true 4K inference with the full 22B model is much heavier than the Wan pipeline.
- On UConn A100 40 GB nodes, the safest first run is not 4K. Start at `1920x1088` or `1536x1024`, verify the pipeline works, then try a higher-resolution profile.
- Do not use the `fp8-cast` quantization preset on A100 nodes. That path is meant for newer architectures and will fail on the A100s currently common in `general-gpu`.
- The Gemma repo is gated on Hugging Face. You must both:
  - accept the Gemma terms on Hugging Face
  - provide a token on HPC, for example `export HF_TOKEN=...`

## Project Layout

- `config/story.example.json`: starter story config
- `config/xianxia_fox_sword_5min_story.json`: long xianxia sample
- `config/xianxia_fox_sword_photoreal_5min_story.json`: long xianxia sample with stronger photoreal identity locking
- `storyboards/xianxia_fox_sword_5min_script.md`: readable version of the same story
- `scripts/download_ltx23_assets.py`: downloads the full model bundle
- `scripts/bootstrap_uconn_hpc.sh`: sets up env + models + optional Slurm submission
- `scripts/story_to_movie.py`: runs LTX scene-by-scene with progress logging
- `slurm/run_story_movie_uconn.slurm`: UConn Storrs batch job

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

The bootstrap script will:

1. install Miniconda if needed
2. create a small bootstrap env with Python 3.10
3. clone/update the official `LTX-2` repo under `vendor/LTX-2`
4. run `uv sync --frozen`
5. download the LTX-2.3 full-model assets plus Gemma
6. optionally submit the Slurm job

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
- `start_image`
- `start_image_strength`
- `enhance_prompt`

## Safer First Runs

Recommended first submit:

- width: `1920`
- height: `1088`
- frames: `121`
- steps: `20` to `30`
- quantization: `null` on A100 nodes

If that works, then try heavier settings. If you later run on Hopper-class GPUs, you can experiment with `fp8-cast`. For a true 4K attempt, set:

- width: `3840`
- height: `2176`

But expect much higher memory pressure and runtime.

## Local Notes

This Mac can scaffold and syntax-check the pipeline wrapper, but it cannot run real LTX inference because the model expects an NVIDIA CUDA GPU.
