#!/usr/bin/env python3
"""Convert a human annotation file (anotazioak_*.json) to the results/emaitzak format.

Input  (list of records):
    {"model": "anthropic_claude-sonnet-5.5", "id": 1, "sentence": ..., "answer": ...,
     "baldintzak": {"1": 1, "2": 0, ...}, "observations": ...}

Output (dict keyed by evaluated model, same as results/emaitzak/<judge>.json):
    {"anthropic/claude-sonnet-5.5": [
        {"id": "1", "baldintzak": {"baldintza_1": 1, "baldintza_2": 0, ...}}, ...]}

Usage:
    python3 1_anotazioak_to_results.py -i anotazioak_2026-10-18_1635.json -o results/emaitzak/
"""
import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict

# Provider prefixes that appear as "provider/model" in the results files
PROVIDERS = ("anthropic", "google", "openai")


def model_key(name, reference_keys):
    """Map an annotation model name to the key used in the results files."""
    if reference_keys:
        for key in reference_keys:
            if key.replace("/", "_") == name:
                return key
        sys.exit(f"Model '{name}' not found in reference keys: {sorted(reference_keys)}")
    provider, sep, rest = name.partition("_")
    if sep and provider in PROVIDERS:
        return f"{provider}/{rest}"
    return name


def convert(annotations, reference_keys=None):
    out = defaultdict(list)
    for rec in annotations:
        key = model_key(rec["model"], reference_keys)
        baldintzak = {
            f"baldintza_{k}": None if v is None else int(v)  # keep unannotated as null
            for k, v in sorted(rec["baldintzak"].items(), key=lambda kv: int(kv[0]))
        }
        out[key].append({"id": str(rec["id"]), "baldintzak": baldintzak})

    # Same ordering as the results files: models in reference order, ids ascending
    for items in out.values():
        items.sort(key=lambda e: int(e["id"]))
    order = list(reference_keys) if reference_keys else sorted(out)
    return {k: out[k] for k in order if k in out}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-i", "--input", required=True, type=Path, help="anotazioak_*.json file")
    parser.add_argument("-o", "--output", type=Path, default=Path("results/emaitzak"),
                        help="output .json file or directory (default: results/emaitzak/)")
    parser.add_argument("-r", "--reference", type=Path,
                        help="a results/emaitzak/*.json file to take model keys and order from")
    args = parser.parse_args()

    if not args.input.is_file():
        parser.error(f"input file not found: {args.input}")
    if args.reference and not args.reference.is_file():
        parser.error(f"reference file not found: {args.reference}")

    # A path without a .json suffix (or an existing directory) is treated as a directory
    output = args.output
    if output.is_dir() or output.suffix != ".json":
        output = output / f"human_{args.input.stem}.json"
    output.parent.mkdir(parents=True, exist_ok=True)

    with open(args.input, encoding="utf-8") as f:
        annotations = json.load(f)

    reference_keys = None
    if args.reference:
        with open(args.reference, encoding="utf-8") as f:
            reference_keys = list(json.load(f).keys())

    result = convert(annotations, reference_keys)

    with open(output, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"Wrote {output}")
    for k, v in result.items():
        print(f"  {k}: {len(v)} items")
    if reference_keys:
        missing = [k for k in reference_keys if k not in result]
        if missing:
            print(f"  Not annotated (present in reference): {missing}")


if __name__ == "__main__":
    main()
