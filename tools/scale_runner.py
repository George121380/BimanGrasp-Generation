#!/usr/bin/env python3
"""
Single-GPU orchestrator to scale up data generation by looping multiple runs
of BimanGrasp-Optimization/main.py until per-object quotas are satisfied.

This script does NOT modify core optimization logic. It wraps main.py with:
- Per-round unique name and seed
- Results reading and incremental merging
- Manifest (resume) management
- Optional cleanup of intermediate round outputs

Usage example:
  conda activate bimangrasp
  python tools/scale_runner.py \
    --exp_name grasp_bigrun \
    --object_code_list Curver_Storage_Bin_Black_Small Hasbro_Monopoly_Hotels_Game \
    --target_count_per_object 10000 \
    --round_batch_size 512 \
    --num_iterations 10000 \
    --seed_base 12345 \
    --resume

Note:
- Results saved by main.py are under data/experiments/<name>/results/
- Each object produces one .npy file named <object_code>.npy (or <object_code>_<step>.npy)
- We aggregate dictionaries list into a final .npz per object.
"""

import argparse
import json
import hashlib
import os
import shutil
import subprocess
import sys
from datetime import datetime
from typing import Any, Dict, List

import numpy as np
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MAIN_ENTRY = os.path.join(REPO_ROOT, 'BimanGrasp-Optimization', 'main.py')


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def experiments_base() -> str:
    # Must match utils.config.PathConfig.experiments_base
    # Default there is '../data/experiments' relative to BimanGrasp-Optimization,
    # but here we operate from repo root: use data/experiments at repo root.
    return os.path.join(REPO_ROOT, 'data', 'experiments')


def results_dir_for_round(exp_name: str, object_code: str, round_id: int, seed: int) -> str:
    name = f"{exp_name}__{object_code}__r{round_id}_s{seed}"
    return os.path.join(experiments_base(), name, 'results')


def manifest_path(exp_name: str) -> str:
    return os.path.join(experiments_base(), exp_name, 'manifest.json')


def final_dir_for_object(exp_name: str, object_code: str, seed_min: int = None, seed_max: int = None) -> str:
    base = os.path.join(experiments_base(), exp_name)
    if seed_min is not None and seed_max is not None:
        return os.path.join(base, f'final_seed_{seed_min}-{seed_max}', object_code)
    return os.path.join(base, 'final', object_code)


def final_np_path(exp_name: str, object_code: str, seed_min: int = None, seed_max: int = None) -> str:
    # Per-user requirement: final file named as obj_code.npy
    return os.path.join(final_dir_for_object(exp_name, object_code, seed_min, seed_max), f'{object_code}.npy')


def load_manifest(exp_name: str) -> Dict[str, Any]:
    mp = manifest_path(exp_name)
    if os.path.isfile(mp):
        with open(mp, 'r') as f:
            return json.load(f)
    return {
        'exp_name': exp_name,
        'objects': {},  # object_code -> {collected:int, rounds:int, seeds:[], done:bool}
        'created_at': datetime.utcnow().isoformat() + 'Z',
        'updated_at': None,
        'version': 1,
    }


def save_manifest(exp_name: str, manifest: Dict[str, Any]) -> None:
    manifest['updated_at'] = datetime.utcnow().isoformat() + 'Z'
    out_dir = os.path.dirname(manifest_path(exp_name))
    ensure_dir(out_dir)
    with open(manifest_path(exp_name), 'w') as f:
        json.dump(manifest, f, indent=2, sort_keys=True)


def compute_seed(seed_base: int, object_index: int, round_id: int) -> int:
    return int(seed_base + object_index * 100_000 + round_id)


def call_main_once(name: str,
                   object_code: str,
                   batch_size: int,
                   num_iterations: int,
                   seed: int,
                   gpu: str = '0',
                   extra_args: List[str] = None,
                   env_overrides: Dict[str, str] = None) -> int:
    """Invoke main.py as a subprocess and return exit code."""
    cmd = [sys.executable, MAIN_ENTRY,
           '--name', name,
           '--object_code', object_code,
           '--batch_size', str(batch_size),
           '--num_iterations', str(num_iterations),
           '--seed', str(seed),
           '--gpu', str(gpu)]
    if extra_args:
        cmd.extend(extra_args)
    env = os.environ.copy()
    if env_overrides:
        env.update(env_overrides)
    # Force per-subprocess GPU isolation so torch sees only the intended GPU.
    # Passing '--gpu' alone is insufficient if torch was imported before setting env vars.
    env['CUDA_VISIBLE_DEVICES'] = str(gpu)
    # Run main.py with CWD set to BimanGrasp-Optimization so relative paths
    # like 'mjcf/right_shadow_hand.xml' resolve correctly.
    proc = subprocess.run(cmd, env=env, cwd=os.path.dirname(MAIN_ENTRY))
    return proc.returncode


def read_round_object_file(results_dir: str, object_code: str) -> List[Dict[str, Any]]:
    """Load list[dict] saved by save_grasp_results for a single object.
    Returns empty list if file missing.
    """
    fpath = os.path.join(results_dir, f'{object_code}.npy')
    if not os.path.isfile(fpath):
        return []
    arr = np.load(fpath, allow_pickle=True)
    if isinstance(arr, np.ndarray):
        return list(arr.tolist())
    return []


def _normalize_for_hash(obj: Any) -> Any:
    """Normalize nested structures for stable hashing.
    - Round floats to 6 decimals
    - Convert numpy scalars
    - Sort dict keys
    """
    try:
        import numpy as _np  # local alias to avoid shadowing
    except Exception:
        _np = None
    if isinstance(obj, dict):
        return {k: _normalize_for_hash(obj[k]) for k in sorted(obj.keys())}
    if isinstance(obj, list):
        return [_normalize_for_hash(v) for v in obj]
    if isinstance(obj, float):
        return round(obj, 6)
    if _np is not None and isinstance(obj, _np.floating):
        return round(float(obj), 6)
    if _np is not None and isinstance(obj, _np.integer):
        return int(obj)
    return obj


def _item_signature(item: Dict[str, Any]) -> str:
    """Compute a stable signature for a sample item.
    Prefer explicit metadata if present; otherwise fall back to content hash.
    """
    meta = item.get('__meta__')
    if isinstance(meta, dict):
        name = str(meta.get('name', ''))
        object_code = str(meta.get('object_code', ''))
        seed = str(meta.get('seed', ''))
        item_index = str(meta.get('item_index', ''))
        return f"{name}|{object_code}|{seed}|{item_index}"
    norm = _normalize_for_hash(item)
    s = json.dumps(norm, sort_keys=True, separators=(',', ':'))
    return hashlib.sha1(s.encode('utf-8')).hexdigest()


def merge_append(current: Dict[str, Any], new_items: List[Dict[str, Any]], target_count: int) -> Dict[str, Any]:
    """Append unique items to current buffers, truncate to target_count.
    Deduplicate by per-item signature to avoid duplicates on resume.
    """
    if current is None:
        current = {'items': []}
    # Build seen signatures set
    seen = set(_item_signature(it) for it in current.get('items', []))
    for it in new_items:
        sig = _item_signature(it)
        if sig in seen:
            continue
        current['items'].append(it)
        seen.add(sig)
        if len(current['items']) >= target_count:
            break
    if len(current['items']) > target_count:
        current['items'] = current['items'][:target_count]
    return current


def _infer_next_round_id(state: Dict[str, Any], collected: int, round_batch_size: int,
                         seed_base: int, object_index: int) -> int:
    """Infer a safe next round_id using multiple signals to avoid duplication after crash.
    Signals: recorded rounds, seeds list, and count-based estimation.
    """
    candidates = [int(state.get('rounds', 0) or 0)]
    seeds = state.get('seeds', []) or []
    if seeds:
        candidates.append(len(seeds))
        base = int(seed_base + object_index * 100_000)
        derived = []
        for s in seeds:
            try:
                rid = int(s) - base
                if rid >= 0:
                    derived.append(rid)
            except Exception:
                pass
        if derived:
            candidates.append(max(derived) + 1)
    if round_batch_size and round_batch_size > 0:
        est = (int(collected) + int(round_batch_size) - 1) // int(round_batch_size)
        candidates.append(est)
    # Ensure non-negative integer
    return max(x for x in candidates if isinstance(x, int) and x >= 0)


def write_final_npz(exp_name: str, object_code: str, buf: Dict[str, Any], seed_min: int = None, seed_max: int = None) -> int:
    out_dir = final_dir_for_object(exp_name, object_code, seed_min, seed_max)
    ensure_dir(out_dir)
    out_path = final_np_path(exp_name, object_code, seed_min, seed_max)
    # Save as .npy with object array of dicts
    np.save(out_path, np.array(buf['items'], dtype=object), allow_pickle=True)
    return len(buf['items'])


def read_final_npz(exp_name: str, object_code: str, seed_min: int = None, seed_max: int = None) -> Dict[str, Any]:
    # Backward-compatible reader: supports both legacy .npz (key 'items') and new .npy
    path = final_np_path(exp_name, object_code, seed_min, seed_max)
    if not os.path.isfile(path):
        # Try legacy .npz
        legacy = os.path.join(final_dir_for_object(exp_name, object_code, seed_min, seed_max), 'final.npz')
        if os.path.isfile(legacy):
            data = np.load(legacy, allow_pickle=True)
            items = list(data['items'].tolist())
            return {'items': items}
        return {'items': []}
    data = np.load(path, allow_pickle=True)
    if hasattr(data, 'files') and 'items' in getattr(data, 'files', []):
        # Legacy .npz
        items = list(data['items'].tolist())
    else:
        # New .npy array of dicts
        items = list(data.tolist()) if isinstance(data, np.ndarray) else []
    return {'items': items}


def cleanup_round(exp_name: str, object_code: str, round_id: int) -> None:
    # Remove the whole experiment directory for that round
    name = f"{exp_name}__{object_code}__r{round_id}"
    exp_dir = os.path.join(experiments_base(), name)
    if os.path.isdir(exp_dir):
        shutil.rmtree(exp_dir, ignore_errors=True)


def list_round_dirs(exp_name: str, object_code: str) -> List[str]:
    base = experiments_base()
    prefix = f"{exp_name}__{object_code}__r"
    if not os.path.isdir(base):
        return []
    paths = []
    for entry in os.listdir(base):
        if entry.startswith(prefix):
            p = os.path.join(base, entry)
            if os.path.isdir(p):
                paths.append(p)
    return paths


def cleanup_all_rounds(exp_name: str, object_code: str) -> None:
    for p in list_round_dirs(exp_name, object_code):
        shutil.rmtree(p, ignore_errors=True)


def copy_final_with_seed_tag(exp_name: str, min_seed: int, max_seed: int) -> str:
    """Create a seed-range-tagged copy of final outputs: final_seed_<min>-<max>/
    Returns the destination directory path.
    """
    exp_root = os.path.join(experiments_base(), exp_name)
    src_final = os.path.join(exp_root, 'final')
    if not os.path.isdir(src_final):
        return ''
    dst_final = os.path.join(exp_root, f'final_seed_{min_seed}-{max_seed}')
    os.makedirs(dst_final, exist_ok=True)
    # Copy per-object dirs/files
    for entry in os.listdir(src_final):
        src_path = os.path.join(src_final, entry)
        dst_path = os.path.join(dst_final, entry)
        if os.path.isdir(src_path):
            shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
        else:
            shutil.copy2(src_path, dst_path)
    return dst_final


def move_final_to_seed_tag(exp_name: str, min_seed: int, max_seed: int) -> str:
    """Move final/ to final_seed_<min>-<max>/ so only seed-tagged dir remains."""
    exp_root = os.path.join(experiments_base(), exp_name)
    src_final = os.path.join(exp_root, 'final')
    dst_final = os.path.join(exp_root, f'final_seed_{min_seed}-{max_seed}')
    if not os.path.isdir(src_final):
        return ''
    # If destination exists, remove it first to allow move
    if os.path.exists(dst_final):
        shutil.rmtree(dst_final, ignore_errors=True)
    os.rename(src_final, dst_final)
    return dst_final


def orchestrate_single_gpu(args: argparse.Namespace) -> None:
    manifest = load_manifest(args.exp_name)

    # Initialize objects state in manifest
    objects_state = manifest['objects']
    for idx, obj in enumerate(args.object_code_list):
        if obj not in objects_state:
            objects_state[obj] = {
                'collected': 0,
                'rounds': 0,
                'seeds': [],
                'done': False,
            }

    save_manifest(args.exp_name, manifest)

    # Iterate objects
    session_seeds: List[int] = []
    for object_index, object_code in enumerate(args.object_code_list):
        state = objects_state[object_code]
        if args.resume and state.get('done', False):
            continue

        # Load current final buffer if any
        final_buf = read_final_npz(args.exp_name, object_code)
        collected = len(final_buf['items'])
        round_id = _infer_next_round_id(state, collected, args.round_batch_size, args.seed_base, object_index)

        while collected < args.target_count_per_object and round_id < args.max_rounds:
            name = f"{args.exp_name}__{object_code}__r{round_id}_s{compute_seed(args.seed_base, object_index, round_id)}"
            seed = compute_seed(args.seed_base, object_index, round_id)
            ret = call_main_once(
                name=name,
                object_code=object_code,
                batch_size=args.round_batch_size,
                num_iterations=args.num_iterations,
                seed=seed,
                gpu=args.gpu,
                extra_args=args.main_extra_args,
            )
            if ret != 0:
                print(f"[WARN] main.py exited with code {ret} for round {name}. Skipping this round.")
                round_id += 1
                continue

            # Read this round output
            rdir = results_dir_for_round(args.exp_name, object_code, round_id, seed)
            items = read_round_object_file(rdir, object_code)
            if not items:
                print(f"[WARN] No items found for round {name}, results missing?")
            else:
                session_seeds.append(seed)
                final_buf = merge_append(final_buf, items, args.target_count_per_object)
                collected = write_final_npz(args.exp_name, object_code, final_buf)

            # Update manifest
            state['collected'] = collected
            state['rounds'] = round_id + 1
            state['seeds'].append(seed)
            state['done'] = collected >= args.target_count_per_object
            save_manifest(args.exp_name, manifest)

            if state['done'] and not args.keep_intermediate:
                # Clean up all historical rounds for this object
                cleanup_all_rounds(args.exp_name, object_code)

            print(f"[INFO] {object_code}: collected={collected}/{args.target_count_per_object}")
            round_id += 1

    # After all objects done: handle final directory mode
    if session_seeds:
        smin, smax = min(session_seeds), max(session_seeds)
        if args.final_dir_mode == 'both':
            dst = copy_final_with_seed_tag(args.exp_name, smin, smax)
            if dst:
                print(f"[INFO] Exported seed-tagged final to: {dst}")
        elif args.final_dir_mode == 'seed_only':
            dst = move_final_to_seed_tag(args.exp_name, smin, smax)
            if dst:
                print(f"[INFO] Moved final to seed-tagged dir: {dst}")
        elif args.final_dir_mode == 'plain_only':
            pass

    print("[OK] Orchestration finished.")


def parse_gpus(gpus_str: str) -> List[str]:
    return [g.strip() for g in gpus_str.split(',') if g.strip()]


def orchestrate_multi_gpu(args: argparse.Namespace) -> None:
    manifest = load_manifest(args.exp_name)

    # Initialize objects state
    objects_state = manifest['objects']
    for idx, obj in enumerate(args.object_code_list):
        if obj not in objects_state:
            objects_state[obj] = {
                'collected': 0,
                'rounds': 0,
                'seeds': [],
                'done': False,
            }

    save_manifest(args.exp_name, manifest)

    # Prepare per-object buffers and scheduling counters
    final_bufs: Dict[str, Dict[str, Any]] = {}
    next_round_id: Dict[str, int] = {}
    inflight_rounds: Dict[str, int] = {}
    for obj in args.object_code_list:
        final_bufs[obj] = read_final_npz(args.exp_name, obj)
        collected = len(final_bufs[obj]['items'])
        next_round_id[obj] = _infer_next_round_id(objects_state[obj], collected, args.round_batch_size, args.seed_base, args.object_code_list.index(obj))
        inflight_rounds[obj] = 0

    gpu_ids = parse_gpus(args.gpus)
    max_workers = len(gpu_ids)

    # Mapping from futures to task metadata
    futures = {}
    gpu_free = [True] * max_workers
    rr_index = 0  # round-robin index over objects

    def schedule_on_gpu(gpu_slot: int) -> bool:
        nonlocal rr_index
        n_objs = len(args.object_code_list)
        for _ in range(n_objs):
            obj = args.object_code_list[rr_index]
            rr_index = (rr_index + 1) % n_objs
            state = objects_state[obj]
            if state.get('done', False):
                continue
            collected = len(final_bufs[obj]['items'])
            remaining_needed = args.target_count_per_object - collected - inflight_rounds[obj] * args.round_batch_size
            if remaining_needed <= 0:
                # Mark as done if not already
                state['done'] = True
                continue
            # Schedule one round for this object
            round_id = next_round_id[obj]
            seed = compute_seed(args.seed_base, args.object_code_list.index(obj), round_id)
            name = f"{args.exp_name}__{obj}__r{round_id}_s{seed}"
            env_overrides = {
                'OMP_NUM_THREADS': '2',
                'MKL_NUM_THREADS': '2',
            }
            # Submit subprocess run in a thread; pass gpu='0' inside isolated env
            future = executor.submit(
                call_main_once,
                name,
                obj,
                args.round_batch_size,
                args.num_iterations,
                seed,
                str(gpu_ids[gpu_slot]),
                args.main_extra_args,
                env_overrides,
            )
            futures[future] = {
                'gpu_slot': gpu_slot,
                'object_code': obj,
                'round_id': round_id,
                'seed': seed,
                'name': name,
            }
            inflight_rounds[obj] += 1
            next_round_id[obj] += 1
            gpu_free[gpu_slot] = False
            return True
        return False

    session_seeds: List[int] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Main loop: schedule until all objects done and no inflight tasks
        while True:
            # Try to schedule on all free GPUs
            scheduled_any = False
            for i in range(max_workers):
                if gpu_free[i]:
                    if schedule_on_gpu(i):
                        scheduled_any = True

            # If nothing scheduled and nothing inflight -> done
            if not scheduled_any and not futures:
                break

            # Wait for at least one to complete
            done, _ = wait(futures.keys(), timeout=None, return_when=FIRST_COMPLETED)
            for fut in done:
                meta = futures.pop(fut)
                gpu_free[meta['gpu_slot']] = True
                obj = meta['object_code']
                round_id = meta['round_id']
                seed = meta['seed']
                ret = fut.result()
                inflight_rounds[obj] = max(0, inflight_rounds[obj] - 1)
                if ret != 0:
                    print(f"[WARN] main.py exited with code {ret} for round {meta['name']}. Skipping this round.")
                else:
                    # Merge results
                    rdir = results_dir_for_round(args.exp_name, obj, round_id, seed)
                    items = read_round_object_file(rdir, obj)
                    if not items:
                        print(f"[WARN] No items found for round {meta['name']}, results missing?")
                    else:
                        session_seeds.append(seed)
                        final_bufs[obj] = merge_append(final_bufs[obj], items, args.target_count_per_object)
                        collected = write_final_npz(args.exp_name, obj, final_bufs[obj])
                        # Update manifest after successful merge
                        state = objects_state[obj]
                        state['collected'] = collected
                        state['rounds'] = state.get('rounds', 0) + 1
                        state.setdefault('seeds', []).append(seed)
                        state['done'] = collected >= args.target_count_per_object
                        save_manifest(args.exp_name, manifest)
                        print(f"[INFO] {obj}: collected={collected}/{args.target_count_per_object}")

        # Final cleanup per object if requested
        if not args.keep_intermediate:
            for obj in args.object_code_list:
                if objects_state[obj].get('done', False):
                    cleanup_all_rounds(args.exp_name, obj)

        # Handle final directory mode
        if session_seeds:
            smin, smax = min(session_seeds), max(session_seeds)
            if args.final_dir_mode == 'both':
                dst = copy_final_with_seed_tag(args.exp_name, smin, smax)
                if dst:
                    print(f"[INFO] Exported seed-tagged final to: {dst}")
            elif args.final_dir_mode == 'seed_only':
                dst = move_final_to_seed_tag(args.exp_name, smin, smax)
                if dst:
                    print(f"[INFO] Moved final to seed-tagged dir: {dst}")
            elif args.final_dir_mode == 'plain_only':
                pass

    print("[OK] Multi-GPU orchestration finished.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Scale-up orchestrator for BimanGrasp-Generation')
    p.add_argument('--exp_name', required=True, type=str, help='Base experiment name')
    p.add_argument('--object_code_list', nargs='+', required=True, help='List of object codes')
    p.add_argument('--target_count_per_object', required=True, type=int, help='Total desired samples per object')
    p.add_argument('--round_batch_size', default=512, type=int, help='Batch size used per round for main.py')
    p.add_argument('--num_iterations', default=10000, type=int, help='Iterations for main.py')
    p.add_argument('--seed_base', default=12345, type=int, help='Seed base used to derive unique seeds')
    p.add_argument('--max_rounds', default=1_000_000, type=int, help='Safety cap on max rounds per object')
    p.add_argument('--gpu', default='0', type=str, help='GPU id visible to main.py (single GPU orchestrator)')
    p.add_argument('--gpus', default=None, type=str, help='Comma-separated GPU ids for multi-GPU orchestration')
    p.add_argument('--resume', action='store_true', help='Resume from manifest/final if present')
    p.add_argument('--final_dir_mode', default='seed_only', choices=['both','seed_only','plain_only'],
                   help='How to export final outputs: both dirs, only seed-tag dir, or only plain final')
    p.add_argument('--keep_intermediate', action='store_true', help='Keep per-round outputs even after quota reached')
    p.add_argument('--main_extra_args', nargs=argparse.REMAINDER, default=[],
                   help='Additional args passed to main.py. Use after -- to avoid parser conflicts.')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    if args.gpus:
        orchestrate_multi_gpu(args)
    else:
        orchestrate_single_gpu(args)


