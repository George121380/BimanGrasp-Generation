#!/usr/bin/env python3
"""
Batch scheduler that reads data/objects_by_category.json and runs
tools/scale_runner.py in sequential batches so that each GPU processes
exactly one object at a time per batch.

This wraps the existing multi-GPU orchestrator to:
- Build the full object list (optionally filtered to categories and to
  existing meshes under --meshroot)
- Split into batches of size == number of GPUs (configurable)
- Launch one scale_runner job per batch and wait until completion

Example:
  conda activate bimangrasp
  python tools/batch_scale_from_json.py \
    --json_path /home/peiqi621/projects/BimanGrasp-Generation/data/objects_by_category.json \
    --meshroot /home/peiqi621/projects/BimanGrasp-Generation/data/meshdata \
    --exp_name cthulhu_all \
    --gpus 0,1,2,3,4,5 \
    --target_count_per_object 30000 \
    --round_batch_size 2048 \
    --num_iterations 10000 \
    --seed_base 0 \
    --resume \
    --final_dir_mode seed_only

Example (uni2bim mode):
  python tools/batch_scale_from_json.py \
    --json_path /home/peiqi621/projects/BimanGrasp-Generation/data/objects_by_category.json \
    --meshroot /home/peiqi621/projects/BimanGrasp-Generation/data/meshdata \
    --exp_name cthulhu_u2b \
    --gpus 0,1,2,3 \
    --target_count_per_object 1024 \
    --round_batch_size 1024 \
    --num_iterations 10000 \
    --mode uni2bim

Notes:
- This script does not change how results are produced; it just sequences
  calls to tools/scale_runner.py. See that script for resume behavior,
  manifest handling, and final directory modes.
"""

import argparse
import os
import subprocess
import sys
from typing import List, Optional


# Ensure repository root is on sys.path so 'tools' can be imported regardless of invocation path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Reuse the JSON loader so category/object existence checks stay consistent
from tools.object_list_from_json import load_object_codes, load_need_add_codes  # type: ignore


SCALE_RUNNER = os.path.join(REPO_ROOT, "tools", "scale_runner.py")


def parse_gpus(gpus_str: str) -> List[str]:
    return [g.strip() for g in gpus_str.split(",") if g.strip()]


def build_batches(items: List[str], batch_size: int) -> List[List[str]]:
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


def run_one_batch(
    exp_name: str,
    gpus: str,
    object_codes: List[str],
    target_count_per_object: int,
    round_batch_size: int,
    num_iterations: int,
    seed_base: int,
    resume: bool,
    final_dir_mode: str,
    keep_intermediate: bool,
    main_extra_args: Optional[List[str]] = None,
) -> int:
    cmd: List[str] = [
        sys.executable,
        SCALE_RUNNER,
        "--exp_name",
        exp_name,
        "--gpus",
        gpus,
        "--object_code_list",
        *object_codes,
        "--target_count_per_object",
        str(target_count_per_object),
        "--round_batch_size",
        str(round_batch_size),
        "--num_iterations",
        str(num_iterations),
        "--seed_base",
        str(seed_base),
        "--final_dir_mode",
        final_dir_mode,
    ]
    if resume:
        cmd.append("--resume")
    if keep_intermediate:
        cmd.append("--keep_intermediate")
    if main_extra_args:
        # Pass through to scale_runner.py using its --main_extra_args (REMAINDER)
        # Do NOT include a bare '--' here; scale_runner will forward these tokens directly to main.py
        cmd.append("--main_extra_args")
        cmd.extend(main_extra_args)

    print(f"[BATCH] Launching {len(object_codes)} objects on GPUs {gpus} -> {object_codes}")
    proc = subprocess.run(cmd, cwd=REPO_ROOT)
    return proc.returncode


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Batch runner over objects_by_category.json or object_need_add.json")
    p.add_argument(
        "--json_path",
        type=str,
        default=os.path.join(REPO_ROOT, "data", "objects_by_category.json"),
        help="Path to objects_by_category.json",
    )
    p.add_argument(
        "--need_add_path",
        type=str,
        default=None,
        help="Optional path to object_need_add.json; when provided, objects are loaded from this file",
    )
    p.add_argument(
        "--meshroot",
        type=str,
        default=os.path.join(REPO_ROOT, "data", "meshdata"),
        help="Mesh root to filter only existing objects",
    )
    p.add_argument("--categories", nargs="+", default=None, help="Optional subset of categories to include")
    p.add_argument(
        "--select_objects",
        nargs="+",
        default=None,
        help="When using --need_add_path, optionally select a subset of objects by exact code",
    )
    p.add_argument(
        "--need_add_all",
        action="store_true",
        help="When using --need_add_path, include all listed items, not only those with valid<threshold",
    )
    p.add_argument("--gpus", required=True, type=str, help="Comma-separated GPU ids, e.g. 0,1,2,3")
    p.add_argument("--exp_name", required=True, type=str, help="Base experiment name")
    p.add_argument("--target_count_per_object", required=True, type=int, help="Quota per object")
    p.add_argument("--round_batch_size", default=512, type=int, help="Batch size per round for main.py")
    p.add_argument("--num_iterations", default=10000, type=int, help="Iterations for main.py")
    p.add_argument("--seed_base", default=0, type=int, help="Seed base used to derive unique seeds")
    p.add_argument(
        "--mode",
        default="default",
        choices=["default", "uni2bim"],
        help="Mode passed to main.py (default|uni2bim). uni2bim decouples left/right optimization.",
    )
    p.add_argument(
        "--final_dir_mode",
        default="seed_only",
        choices=["both", "seed_only", "plain_only"],
        help="How to export final outputs for each batch session",
    )
    p.add_argument("--keep_intermediate", action="store_true", help="Keep per-round outputs after quota reached")
    p.add_argument("--resume", action="store_true", help="Resume behavior for each batch invocation")
    p.add_argument(
        "--items_per_batch",
        type=int,
        default=None,
        help="Override items per batch (defaults to number of GPUs)",
    )
    p.add_argument("--shuffle", action="store_true", help="Shuffle object order before batching")
    p.add_argument("--start_index", type=int, default=0, help="Start index into flattened object list")
    p.add_argument("--max_items", type=int, default=None, help="Optional cap on number of objects to process")
    p.add_argument("--dry_run", action="store_true", help="Print plan and exit without running")
    p.add_argument("--", dest="main_extra_args", nargs=argparse.REMAINDER, help=argparse.SUPPRESS)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Choose source: need_add or category json
    if args.need_add_path:
        all_codes = load_need_add_codes(
            json_path=args.need_add_path,
            meshroot=args.meshroot,
            categories=args.categories,
            objects=args.select_objects,
            include_all=args.need_add_all,
            dedup=True,
        )
    else:
        all_codes = load_object_codes(
            json_path=args.json_path,
            meshroot=args.meshroot,
            categories=args.categories,
            dedup=True,
        )

    # Slice and optionally shuffle
    codes = all_codes[args.start_index :]
    if args.max_items is not None:
        codes = codes[: args.max_items]
    if args.shuffle:
        import random

        random.shuffle(codes)

    gpu_list = parse_gpus(args.gpus)
    if not gpu_list:
        print("[ERROR] No GPUs provided via --gpus")
        sys.exit(2)

    items_per_batch = args.items_per_batch or len(gpu_list)
    if items_per_batch <= 0:
        print("[ERROR] items_per_batch must be positive")
        sys.exit(2)

    batches = build_batches(codes, items_per_batch)

    print(f"[PLAN] Total objects: {len(codes)} | GPUs: {len(gpu_list)} | items_per_batch: {items_per_batch} | batches: {len(batches)}")
    if args.dry_run:
        for bi, b in enumerate(batches):
            print(f"  - Batch {bi+1}/{len(batches)}: {b}")
        return

    # Guard against destructive 'seed_only' behavior across multiple batches.
    # When scale_runner is invoked with final_dir_mode=seed_only, it renames
    # 'final/' to 'final_seed_<min>-<max>/'. If multiple invocations produce the
    # same seed range (common when seeds depend only on per-batch object indices),
    # later batches can overwrite earlier ones. To avoid data loss, auto-upgrade
    # to 'both' when running more than one batch.
    runner_final_dir_mode = args.final_dir_mode
    if runner_final_dir_mode == "seed_only" and len(batches) > 1:
        print("[WARN] Multiple batches detected with final_dir_mode=seed_only. "
              "This may overwrite previous batches. Switching to final_dir_mode=both for safety.")
        runner_final_dir_mode = "both"

    # Prepare extra args for main.py, merging explicit --mode when provided
    runner_main_extra_args = list(args.main_extra_args) if args.main_extra_args else []
    if args.mode and args.mode != "default":
        if "--mode" not in runner_main_extra_args:
            runner_main_extra_args.extend(["--mode", args.mode])

    # Run batches sequentially
    for bi, batch_codes in enumerate(batches):
        print(f"[RUN] Batch {bi+1}/{len(batches)} starting with {len(batch_codes)} objects")
        ret = run_one_batch(
            exp_name=args.exp_name,
            gpus=args.gpus,
            object_codes=batch_codes,
            target_count_per_object=args.target_count_per_object,
            round_batch_size=args.round_batch_size,
            num_iterations=args.num_iterations,
            seed_base=args.seed_base,
            resume=args.resume,
            final_dir_mode=runner_final_dir_mode,
            keep_intermediate=args.keep_intermediate,
            main_extra_args=runner_main_extra_args,
        )
        if ret != 0:
            print(f"[WARN] Batch {bi+1} exited with code {ret}. Continuing to next batch.")

    print("[OK] All batches completed.")


if __name__ == "__main__":
    main()


