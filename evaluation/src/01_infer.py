import argparse
import json
import os
import sys
import time
from pathlib import Path

import requests
from tqdm import tqdm

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


def query_openrouter(messages, model, api_key, max_retries=3):
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.post(
                OPENROUTER_URL,
                headers={"Authorization": f"Bearer {api_key}"},
                json={"model": model, "messages": messages, "reasoning": {"effort": "high"}},
                timeout=900,
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except (requests.RequestException, KeyError, IndexError) as e:
            if attempt == max_retries:
                tqdm.write(f"Failed after {max_retries} attempts: {e}")
                return None
            time.sleep(2 ** attempt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-key", default=os.environ.get("OPENROUTER_API_KEY"))
    args = parser.parse_args()

    # Resume from the output if it exists: rows without a response are (re)queried
    path = args.output_path if args.output_path.exists() else args.input_path
    rows = json.loads(path.read_text(encoding="utf-8"))
    args.output_path.parent.mkdir(parents=True, exist_ok=True)

    n_errors = 0
    for row in tqdm([row for row in rows if not row.get("infer", {}).get("infer_response")], desc=args.model):
        messages = [{"role": "user", "content": row["question"]}]
        response = query_openrouter(messages, args.model, args.api_key)
        if not response:
            n_errors += 1
            continue
        row["infer"] = {"infer_model": args.model, "infer_user_message": row["question"], "infer_response": response}
        # Save after every row so an interruption keeps the progress
        args.output_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"{args.output_path}: {n_errors} errors (rerun to retry them)")
    sys.exit(1 if n_errors else 0)
