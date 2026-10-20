"""Group judge results: one output file per judge model.

Output structure (results/emaitzak/<judge_model>.json):
{
  "<infer_model>": [
    {"id": "1", "baldintzak": {"baldintza_1": 1, ...}},
    ...
  ],
  ...
}

Usage:
    python3 0_group_by_judge.py -i results/ -o results/emaitzak/
"""
import argparse
import glob
import json
import os
from collections import defaultdict

parser = argparse.ArgumentParser(description="Group judge results into one file per judge model.")
parser.add_argument("-i", "--input-dir", default=".", help="Directory with the input JSON files (default: current directory)")
parser.add_argument("-o", "--output-dir", default="results/emaitzak", help="Directory to write the per-judge files (default: results/emaitzak)")
args = parser.parse_args()

INPUT_DIR = args.input_dir
OUTPUT_DIR = args.output_dir

# judge_model -> infer_model -> list of {id, baldintzak}
grouped = defaultdict(lambda: defaultdict(list))

for path in sorted(glob.glob(os.path.join(INPUT_DIR, "*.json"))):
    with open(path, encoding="utf-8") as f:
        records = json.load(f)
    for rec in records:
        infer_model = rec.get("infer", {}).get("infer_model")
        judge_model = rec.get("judge", {}).get("judge_model")
        if infer_model is None or judge_model is None:
            print(f"Skipping record {rec.get('id')} in {os.path.basename(path)}: missing model name")
            continue
        grouped[judge_model][infer_model].append({
            "id": rec.get("id"),
            "baldintzak": (rec.get("result") or {}).get("baldintzak"),
        })

os.makedirs(OUTPUT_DIR, exist_ok=True)
for judge_model, by_infer in grouped.items():
    out_path = os.path.join(OUTPUT_DIR, judge_model.replace("/", "_") + ".json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(by_infer, f, ensure_ascii=False, indent=2)
    counts = ", ".join(f"{m}: {len(v)}" for m, v in by_infer.items())
    print(f"{out_path} -> {counts}")
