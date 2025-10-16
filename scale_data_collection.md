## Scale Data Collection Guide (tools/scale_runner.py)

This guide explains how to run large-scale data generation using the external orchestrator script `tools/scale_runner.py`. It loops `BimanGrasp-Optimization/main.py` multiple times until each object reaches a target sample quota, supports resume, merging, cleanup, and multi-GPU parallelism. All code comments and CLI docs are in English; usage instructions here are concise.

---

### Key Features
- Per-object quota: keep generating in rounds until target count is reached
- Deterministic seeds: unique per object and round
- Result merging into a single final file per object
- Resume from manifest (safe to rerun partially completed runs)
- Cleanup of intermediate round outputs when done
- Multi-GPU parallel execution with per-process GPU binding
- Seed-range tagged final directory naming

---

### Seed Strategy
Seed for object index `i` and round `r`:
- `seed = seed_base + i * 100000 + r`
This ensures no collisions across objects and rounds. The round experiment name includes the seed: `<exp_name>__<object>__r<r>_s<seed>`.

---

### Output Layout
- Per-round results: `data/experiments/<exp_name>__<object>__r<round>_s<seed>/results/<object>.npy`
- Final (seed-tagged) results: `data/experiments/<exp_name>/final_seed_<minseed>-<maxseed>/<object>/<object>.npy`
- Manifest: `data/experiments/<exp_name>/manifest.json`

Note: By default, only the seed-tagged final directory is kept (`--final_dir_mode seed_only`).

---

### CLI Arguments
Required:
- `--exp_name <str>`: Base experiment name.
- `--object_code_list <obj1> <obj2> ...`: Space-separated list of object codes.
- `--target_count_per_object <int>`: Total desired samples per object.

Core parameters:
- `--round_batch_size <int>`: Batch size for each round (passed to main.py `--batch_size`).
- `--num_iterations <int>`: Iterations for each round (passed to main.py `--num_iterations`).
- `--seed_base <int>`: Base seed; actual seed computed per object+round.
- `--max_rounds <int>`: Safety cap on total rounds per object (default: 1,000,000).

GPU / Performance:
- Single GPU:
  - `--gpu <id>`: GPU id for single-GPU mode (e.g., `0`).
- Multi-GPU:
  - `--gpus <ids>`: Comma-separated GPU ids for multi-GPU mode (e.g., `0,1,2`).
  - Each subprocess is bound to the specified GPU id via `--gpu <id>`.

Resume / Cleanup / Final Dir:
- `--resume`: Continue from `manifest.json` and any existing final files.
- `--keep_intermediate`: Keep per-round outputs even after quota is met (default: clean all rounds per object when done).
- `--final_dir_mode {seed_only|both|plain_only}`: Final output directory mode.
  - `seed_only` (default): Only `final_seed_<min>-<max>/`.
  - `both`: Keep both `final/` and `final_seed_<min>-<max>/`.
  - `plain_only`: Only `final/`.

Pass-through args to main.py:
- `--main_extra_args -- <args...>`: Anything after `--` is forwarded to `main.py`. Examples: `--vis`, `--vis_frame_stride 200`.

---

### Examples
Single GPU (basic):
```bash
conda activate bimangrasp
cd /home/peiqi621/projects/BimanGrasp-Generation
python tools/scale_runner.py \
  --exp_name grasp_bigrun \
  --object_code_list Curver_Storage_Bin_Black_Small Hasbro_Monopoly_Hotels_Game \
  --target_count_per_object 10000 \
  --round_batch_size 512 \
  --num_iterations 10000 \
  --seed_base 12345 \
  --gpu 0 \
  --resume
```

Single GPU with visualization pass-through:
```bash
python tools/scale_runner.py \
  --exp_name vis_run \
  --object_code_list merged_collision_300k_wt \
  --target_count_per_object 512 \
  --round_batch_size 64 \
  --num_iterations 2000 \
  --seed_base 7 \
  --gpu 0 \
  --resume \
  --final_dir_mode seed_only \
  -- --vis --vis_frame_stride 200 --vis_fps 20
```

Multi-GPU (2 GPUs):
```bash
python tools/scale_runner.py \
  --exp_name mgpu_run \
  --gpus 0,1 \
  --object_code_list merged_collision_300k_wt Curver_Storage_Bin_Black_Small \
  --target_count_per_object 8000 \
  --round_batch_size 256 \
  --num_iterations 10000 \
  --seed_base 777 \
  --resume \
  --final_dir_mode seed_only
```

Keep both plain and seed-tagged final directories:
```bash
python tools/scale_runner.py \
  --exp_name dual_final \
  --gpus 0,1,2 \
  --object_code_list Curver_Storage_Bin_Black_Small \
  --target_count_per_object 4096 \
  --round_batch_size 256 \
  --num_iterations 8000 \
  --seed_base 9000 \
  --resume \
  --final_dir_mode both
```

Keep intermediate round directories for debugging:
```bash
python tools/scale_runner.py \
  --exp_name debug_rounds \
  --object_code_list merged_collision_300k_wt \
  --target_count_per_object 256 \
  --round_batch_size 64 \
  --num_iterations 1000 \
  --seed_base 123 \
  --gpu 0 \
  --resume \
  --keep_intermediate
```

---

### Notes and Tips
- If `nvidia-smi` shows only GPU0 being utilized, ensure no global `CUDA_VISIBLE_DEVICES` is restricting visibility. The orchestrator passes `--gpu <id>` directly to `main.py` per subprocess.
- For I/O heavy runs (visualization on), consider larger `--vis_frame_stride` and smaller `--vis_width/--vis_height` passed via `--main_extra_args`.
- Merged final file format: per-object `.npy` storing an object-array of dicts (same keys as per-round output). Legacy `.npz` (`final.npz`) is still readable for backward compatibility.


