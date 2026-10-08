import argparse
import importlib
import json
import sys
import time
from pathlib import Path

from openai import OpenAI
from tqdm import tqdm

# Module names starting with a digit can't be imported with a plain import statement
infer_latxa = importlib.import_module("01_infer_latxa")
parse_judge_response = importlib.import_module("03_judge").parse_judge_response

# The judge only outputs a short JSON; the 32768 used for inference doesn't fit Latxa's 65536 context with long prompts
JUDGE_MAX_TOKENS = 4096


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", type=Path, required=True, help="JSON written by 02_preprocess_judge.py")
    parser.add_argument("--output-path", type=Path, required=True)
    args = parser.parse_args()

    client = OpenAI(base_url=infer_latxa.LATXA_URL, api_key=infer_latxa.LATXA_API_KEY, timeout=600)

    # Resume from the output if it exists: rows without a result are (re)queried
    path = args.output_path if args.output_path.exists() else args.input_path
    rows = json.loads(path.read_text(encoding="utf-8"))
    args.output_path.parent.mkdir(parents=True, exist_ok=True)

    # Rows without a judge prompt (failed inference) can't be judged, rerun the inference and 02_preprocess_judge.py first
    missing = [row["id"] for row in rows if "judge" not in row]
    if missing:
        print(f"Skipping rows without a judge prompt: {missing}")

    n_errors = 0
    for row in tqdm([row for row in rows if "judge" in row and not row.get("result")], desc=infer_latxa.LATXA_MODEL):
        # The judge system message replaces Latxa's default one in the custom chat template
        messages = [
            {"role": "system", "content": row["judge"]["judge_system_message"]},
            {"role": "user", "content": row["judge"]["judge_user_message"]},
        ]
        response = infer_latxa.query_latxa(client, messages, max_tokens=JUDGE_MAX_TOKENS)
        time.sleep(infer_latxa.SLEEP_BETWEEN_QUERIES)
        if not response:
            n_errors += 1
            continue
        result = parse_judge_response(response)
        if result is None:
            tqdm.write(f"Row {row['id']}: no valid JSON in the judge response: {response!r}")
            n_errors += 1
            continue
        row["judge"] = {"judge_model": infer_latxa.LATXA_MODEL, **row["judge"]}
        row["result"] = result
        # Save after every row so an interruption keeps the progress
        args.output_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"{args.output_path}: {n_errors} errors (rerun to retry them)")
    sys.exit(1 if n_errors else 0)
