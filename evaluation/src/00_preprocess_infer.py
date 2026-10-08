import argparse
import json
# import sys
from pathlib import Path

import pandas as pd


def main(input_path: Path, output_path: Path = None, overwrite=False):

    if output_path is None:
        output_path = input_path.with_name(f"{input_path.stem}.processed.json")
    # The first row holds the prompt templates, it is saved apart from the data
    template_path = output_path.with_name(f"{output_path.stem}.template.json")
    for path in (output_path, template_path):
        assert not path.exists() or overwrite, f"Output file {path} already exists. Use --overwrite to overwrite it."

    old_col = "raw_question"
    # Read as text (keeps empty cells as "" instead of NaN) and strip a possible BOM
    df: pd.DataFrame = pd.read_csv(input_path, header=0, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    assert not old_col in df.columns or overwrite, f"File {input_path} has already been preprocessed. Use --overwrite to overwrite it."

    template = df["question"].iloc[0]
    df[old_col] = df["question"]
    df.loc[1:, "question"] = df.loc[1:, old_col].apply(lambda q: template.format(question=q))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with template_path.open("w", encoding="utf-8") as f:
        json.dump(df.drop(columns=old_col).iloc[0].to_dict(), f, ensure_ascii=False, indent=2)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(df.iloc[1:].to_dict(orient="records"), f, ensure_ascii=False, indent=2)


if __name__ == "__main__":

    # sys.argv += [
    #     "--path", "/home/nperez/Development/metalinguistics/experiments/01-dev-01062026/inputs/zuzena_okerra.csv"
    #  ]

    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=False, help="Defaults to <input>.processed.json (templates go to <output>.template.json)")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    main(args.path, args.output_path, overwrite=args.overwrite)