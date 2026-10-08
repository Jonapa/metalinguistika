import argparse
import json
from pathlib import Path


def to_bullets(s: str) -> str:
    if not s.strip():
        return ""
    return "\n   * " + "\n   * ".join([i.strip().replace("“", "\"").replace("”", "\"") for i in s.split("\n") if i.strip()])


def instantiate_prompt(row, correct_prompt, incorrect_prompt) -> str:
    template = correct_prompt if row["answer"] == "1" else incorrect_prompt
    fields = {**row, **{k: to_bullets(row[k]) for k in ("rule", "conflict", "correction", "glossary")}, "response": row["infer"]["infer_response"]}
    prompt = template.format(**fields)
    if not fields["glossary"] and "=== GLOSATEGIA ===" in prompt:
        # Drop only the glossary section, the output format after it must be kept
        before, after = prompt.split("=== GLOSATEGIA ===")
        prompt = before + "=== IRTEERA-FORMATUA ===" + after.split("=== IRTEERA-FORMATUA ===")[1]
        prompt = prompt.replace(", baldintza-zerrenda bat eta glosategi bat emango zaizkizu.", " eta baldintza-zerrenda bat emango zaizkizu.")
        prompt = prompt.replace("* Onartu glosategian zehaztutako termino sinonimoak parafrasiak eta formulazio baliokideak interpretatzeko.\n", "")
    return prompt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", type=Path, required=True, help="JSON written by 01_infer.py")
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--correct-prompt", type=Path, required=True, help="Judge prompt for correct sentences (answer 1)")
    parser.add_argument("--incorrect-prompt", type=Path, required=True, help="Judge prompt for incorrect sentences (answer 0)")
    parser.add_argument("--system-prompt", type=Path, required=True, help="Judge system message, saved as judge.judge_system_message")
    args = parser.parse_args()

    correct_prompt = args.correct_prompt.read_text(encoding="utf-8")
    incorrect_prompt = args.incorrect_prompt.read_text(encoding="utf-8")
    system_prompt = args.system_prompt.read_text(encoding="utf-8").strip()
    rows = json.loads(args.input_path.read_text(encoding="utf-8"))

    # Rows whose inference failed have no response to judge, rerun 01_infer.py first
    missing = [row["id"] for row in rows if not row.get("infer", {}).get("infer_response")]
    if missing:
        print(f"Skipping rows without a response: {missing}")
    for row in rows:
        if row.get("infer", {}).get("infer_response"):
            row["judge"] = {
                "judge_system_message": system_prompt,
                "judge_user_message": instantiate_prompt(row, correct_prompt, incorrect_prompt),
            }

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved to {args.output_path}")
