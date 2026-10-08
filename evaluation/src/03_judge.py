import argparse
import importlib
import json
import os
import re
import sys
from pathlib import Path

from tqdm import tqdm

# Module names starting with a digit can't be imported with a plain import statement
query_openrouter = importlib.import_module("01_infer").query_openrouter


def parse_judge_response(response: str):
    # Judges sometimes wrap the JSON in a markdown block or add text around it, keep the outermost {...}
    match = re.search(r"\{.*\}", response, re.DOTALL)
    if not match:
        return None
    # The incorrect prompt's output example has a trailing comma, remove it if the judge copied it
    text = re.sub(r",\s*([}\]])", r"\1", match.group(0))
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", type=Path, required=True, help="JSON written by 02_preprocess_judge.py")
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--model", required=True, help="OpenRouter judge model")
    parser.add_argument("--api-key", default=os.environ.get("OPENROUTER_API_KEY"))
    args = parser.parse_args()
    if not args.api_key:
        parser.error("no API key: pass --api-key or set OPENROUTER_API_KEY")

    # Resume from the output if it exists: rows without a result are (re)queried
    path = args.output_path if args.output_path.exists() else args.input_path
    rows = json.loads(path.read_text(encoding="utf-8"))
    args.output_path.parent.mkdir(parents=True, exist_ok=True)

    # Rows without a judge prompt (failed inference) can't be judged, rerun 01_infer.py and 02_preprocess_judge.py first
    missing = [row["id"] for row in rows if "judge" not in row]
    if missing:
        print(f"Skipping rows without a judge prompt: {missing}")

    n_errors = 0
    for row in tqdm([row for row in rows if "judge" in row and not row.get("result")], desc=args.model):
        messages = [
            {"role": "system", "content": row["judge"]["judge_system_message"]},
            {"role": "user", "content": row["judge"]["judge_user_message"]},
        ]
        response = query_openrouter(messages, args.model, args.api_key)
        if not response:
            n_errors += 1
            continue
        result = parse_judge_response(response)
        if result is None:
            tqdm.write(f"Row {row['id']}: no valid JSON in the judge response: {response!r}")
            n_errors += 1
            continue
        row["judge"] = {"judge_model": args.model, **row["judge"]}
        row["result"] = result
        # Save after every row so an interruption keeps the progress
        args.output_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"{args.output_path}: {n_errors} errors (rerun to retry them)")
    sys.exit(1 if n_errors else 0)
