#!/usr/bin/env python3
"""
Utility to flatten data/objects_by_category.json into a space-separated list of
object codes suitable for passing to tools/scale_runner.py via --object_code_list.

Example:
  python tools/object_list_from_json.py \
    /home/peiqi621/projects/BimanGrasp-Generation/data/objects_by_category.json \
    --meshroot /home/peiqi621/projects/BimanGrasp-Generation/data/meshdata

Then use command substitution:
  python tools/scale_runner.py --object_code_list $(python tools/object_list_from_json.py ...)
"""

import argparse
import json
import os
import sys
from typing import Iterable, List, Optional, Set, Dict, Any


def _deduplicate_preserve_order(items: Iterable[str]) -> List[str]:
    seen: Set[str] = set()
    out: List[str] = []
    for it in items:
        if it not in seen:
            seen.add(it)
            out.append(it)
    return out


def _category_exists(meshroot: str, category: str) -> bool:
    return os.path.isdir(os.path.join(meshroot, category))


def _object_exists(meshroot: str, category: str, code: str) -> bool:
    return os.path.isdir(os.path.join(meshroot, category, code))


def load_object_codes(
    json_path: str,
    meshroot: Optional[str] = None,
    categories: Optional[List[str]] = None,
    dedup: bool = True,
) -> List[str]:
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError("JSON must be an object mapping categories to lists of codes")

    obj_codes: List[str] = []
    wanted_categories = set(categories) if categories else None

    for cat, entries in data.items():
        if wanted_categories is not None and cat not in wanted_categories:
            continue
        if not isinstance(entries, list):
            continue

        if meshroot and not _category_exists(meshroot, cat):
            # Skip categories not present under meshroot
            continue

        for entry in entries:
            code = str(entry)
            if meshroot and not _object_exists(meshroot, cat, code):
                # Skip missing objects when meshroot is provided
                continue
            obj_codes.append(code)

    return _deduplicate_preserve_order(obj_codes) if dedup else obj_codes


def load_need_add_codes(
    json_path: str,
    meshroot: Optional[str] = None,
    categories: Optional[List[str]] = None,
    objects: Optional[List[str]] = None,
    include_all: bool = False,
    dedup: bool = True,
) -> List[str]:
    """Load object codes from an object_need_add.json file.

    The JSON structure is expected as:
      {"threshold": int, "count": int, "items": [{"category": str, "object": str, "valid": int}, ...]}

    Filtering rules:
    - If categories is provided, only include items whose category is in the list.
    - If objects is provided, only include items whose object is in the list.
    - If include_all is False, include only items with valid < threshold.
    - If meshroot is provided, include only ones that exist under meshroot.
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data: Dict[str, Any] = json.load(f)

    items = data.get('items', []) or []
    threshold = int(data.get('threshold', 0) or 0)

    wanted_categories = set(categories) if categories else None
    wanted_objects = set(str(o) for o in objects) if objects else None

    out: List[str] = []
    for it in items:
        try:
            cat = str(it.get('category'))
            code = str(it.get('object'))
            valid = int(it.get('valid', 0) or 0)
        except Exception:
            continue

        if wanted_categories is not None and cat not in wanted_categories:
            continue
        if wanted_objects is not None and code not in wanted_objects:
            continue
        if not include_all and not (valid < threshold):
            continue
        if meshroot and not _object_exists(meshroot, cat, code):
            continue
        out.append(code)

    return _deduplicate_preserve_order(out) if dedup else out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Flatten objects_by_category.json to object codes")
    p.add_argument("json_path", type=str, help="Path to objects_by_category.json")
    p.add_argument("--meshroot", type=str, default=None, help="Mesh root (e.g., data/meshdata) to filter existing only")
    p.add_argument("--categories", nargs='+', default=None, help="Optional subset of categories to include")
    p.add_argument("--no_dedup", action='store_true', help="Disable deduplication of codes")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    codes = load_object_codes(
        json_path=args.json_path,
        meshroot=args.meshroot,
        categories=args.categories,
        dedup=not args.no_dedup,
    )
    # Print as space-separated for easy shell substitution
    sys.stdout.write(" ".join(codes))


if __name__ == "__main__":
    main()


