import argparse
import json
import math

import numpy as np
from datasets import get_dataset_config_names, load_dataset


class RandomPermutation:
    """Lazy pseudo-random permutation of range(n) with O(1) memory.

    Uses a keyed Feistel network with cycle-walking, so every index in [0, n)
    is produced exactly once (no repetitions) without materialising the
    whole permutation.
    """

    def __init__(self, n, seed, rounds=4):
        self.n = n
        bits = max(2, math.ceil(math.log2(max(n, 2))))
        self.half = (bits + 1) // 2
        self.mask = np.uint64((1 << self.half) - 1)
        rng = np.random.default_rng(seed)
        self.keys = rng.integers(0, 2**63, size=rounds, dtype=np.uint64)

    def _mix(self, x, key):
        # splitmix64-style finaliser (uint64 overflow wraps intentionally)
        x = x ^ key
        x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return (x ^ (x >> np.uint64(31))) & self.mask

    def _feistel(self, x):
        h = np.uint64(self.half)
        left, right = x >> h, x & self.mask
        for key in self.keys:
            left, right = right, left ^ self._mix(right, key)
        return (left << h) | right

    def __call__(self, idx):
        x = self._feistel(np.asarray(idx, dtype=np.uint64))
        out_of_range = x >= self.n
        while out_of_range.any():  # cycle-walk back into [0, n)
            x[out_of_range] = self._feistel(x[out_of_range])
            out_of_range = x >= self.n
        return x.astype(np.int64)

    def batches(self, batch_size):
        for start in range(0, self.n, batch_size):
            yield self(np.arange(start, min(start + batch_size, self.n)))


class SplitsView:
    """Read-only view of several datasets as if they were concatenated.

    concatenate_datasets requires identical features, but configurations of the
    same corpus may differ (e.g. `meta` as struct in one and Json in another).
    Global indices are mapped to (split, local index) instead, so no data is copied.
    """

    def __init__(self, splits):
        self.splits = splits
        self.sizes = [len(split) for split in splits]
        self.offsets = np.concatenate([[0], np.cumsum(self.sizes)])

    def __len__(self):
        return int(self.offsets[-1])

    def locate(self, indices):
        indices = np.asarray(indices, dtype=np.int64)
        split_ids = np.searchsorted(self.offsets[1:], indices, side="right")
        return split_ids, indices - self.offsets[split_ids]

    def _gather(self, indices, fetch):
        # Group indices by split, fetch each group, then restore the original order
        split_ids, local = self.locate(indices)
        out = [None] * len(split_ids)
        for s in np.unique(split_ids):
            positions = np.flatnonzero(split_ids == s)
            for pos, value in zip(positions, fetch(self.splits[s].select(local[positions]))):
                out[pos] = value
        return out

    def column(self, indices, field):
        # datasets>=4 returns a lazy Column, not a list
        return self._gather(indices, lambda sub: list(sub[field]))

    def to_json(self, indices, path, batch_size=10_000):
        with open(path, "w", encoding="utf-8") as f:
            for start in range(0, len(indices), batch_size):
                for row in self._gather(indices[start:start + batch_size], lambda sub: iter(sub)):
                    f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


BATCH_SIZE = 1000  # rows tokenized per step


def parse_args():
    parser = argparse.ArgumentParser(description="Randomly sample the train split of a HF dataset.")
    parser.add_argument("--mode", choices=["rows", "tokens"], required=True)
    parser.add_argument("-n", "--num", type=int, required=True, help="Number of rows or tokens")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--text-field", required=True)
    parser.add_argument("--seq-len", type=int, required=True,
                        help="Skip entries longer than this many tokens")
    parser.add_argument("--report", required=True, help="Path of the report file")
    return parser.parse_args()


def load_train(dataset_name):
    configs = get_dataset_config_names(dataset_name)
    print(f"Found configurations: {configs}")

    train_splits = []
    for config in configs:
        print(f"Downloading/Loading train split for: {config}...")
        # Non-streaming load: data is converted once to Arrow on disk and memory-mapped,
        # so random row access is cheap and nothing is loaded into RAM.
        train_splits.append(load_dataset(dataset_name, name=config, split="train"))

    ds = SplitsView(train_splits)
    return ds, configs, ds.sizes


def to_text(value, tokenizer):
    # Chat datasets store a list of {"role", "content"} messages: render them with the
    # chat template (special tokens included) so lengths match the training format
    if isinstance(value, list):
        return tokenizer.apply_chat_template(value, tokenize=False) if value else ""
    return value or ""  # guard against null texts


def batch_lengths(ds, tokenizer, text_field, batch_idx):
    texts = [to_text(v, tokenizer) for v in ds.column(batch_idx, text_field)]
    return [len(ids) for ids in tokenizer(texts, add_special_tokens=False)["input_ids"]]


def sample(ds, perm, mode, num, tokenizer, text_field, max_len):
    # Draw rows in permutation order, skipping any longer than max_len tokens,
    # until num rows (mode "rows") or num tokens (mode "tokens") are collected
    selected, lengths, skipped, total_tokens = [], [], [], 0

    def result():
        return (np.array(selected, dtype=np.int64), np.array(lengths, dtype=np.int64),
                np.array(skipped, dtype=np.int64))

    for batch_idx in perm.batches(BATCH_SIZE):
        for idx, length in zip(batch_idx, batch_lengths(ds, tokenizer, text_field, batch_idx)):
            if length > max_len:
                skipped.append(idx)
                continue
            selected.append(idx)
            lengths.append(length)
            total_tokens += length
            if (total_tokens if mode == "tokens" else len(selected)) >= num:
                print(f"Reached {total_tokens} tokens with {len(selected)} rows "
                      f"({len(skipped)} skipped)")
                return result()
        print(f"  {len(selected)} rows, {total_tokens} tokens ({len(skipped)} skipped)")

    print(f"Corpus exhausted: {total_tokens} tokens with {len(selected)} rows "
          f"({len(skipped)} skipped)")
    return result()


def write_report(args, total_rows, configs, sizes, indices, lengths, skipped):
    def skip_pct(n_kept, n_skipped):
        examined = n_kept + n_skipped
        return 100 * n_skipped / examined if examined else 0.0

    n, tokens = len(lengths), int(lengths.sum())
    lines = [
        "# Sampling report",
        "",
        "## Parameters",
        f"- dataset: {args.dataset} (train split)",
        f"- mode: {args.mode}",
        f"- requested: {args.num} {args.mode}",
        f"- seed: {args.seed}",
        f"- tokenizer: {args.tokenizer}",
        f"- text field: {args.text_field}",
        f"- max tokens per entry: {args.seq_len} (longer entries skipped)",
        f"- output: {args.output}",
        "",
        "## Outcome",
        f"- corpus rows (train): {total_rows}",
        f"- sampled rows: {n} ({100 * n / total_rows:.4f}% of corpus)",
        f"- sampled tokens: {tokens}",
        f"- skipped entries longer than {args.seq_len} tokens: {len(skipped)} "
        f"({skip_pct(n, len(skipped)):.2f}% of examined)",
    ]
    if n:
        lines += [
            f"- tokens per entry: min {lengths.min()}, mean {lengths.mean():.1f}, "
            f"median {int(np.median(lengths))}, max {lengths.max()}",
        ]

    # Map each sampled global index back to the configuration it came from
    bounds = np.cumsum(sizes)
    source = np.searchsorted(bounds, indices, side="right")
    skipped_source = np.searchsorted(bounds, skipped, side="right")
    lines += [
        "",
        "## Per source",
        f"| source | corpus rows | sampled rows | sampled tokens | skipped (> {args.seq_len} tokens) |",
        "|---|---|---|---|---|",
    ]
    for i, (config, size) in enumerate(zip(configs, sizes)):
        lens_i = lengths[source == i]
        n_i, skipped_i = len(lens_i), int((skipped_source == i).sum())
        lines.append(f"| {config} | {size} | {n_i} | {int(lens_i.sum())} | "
                     f"{skipped_i} ({skip_pct(n_i, skipped_i):.2f}%) |")

    with open(args.report, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Report written to {args.report}")


def main():
    from transformers import AutoTokenizer

    args = parse_args()
    ds, configs, sizes = load_train(args.dataset)
    total_rows = len(ds)
    print(f"Train split has {total_rows} rows")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    perm = RandomPermutation(total_rows, args.seed)

    if args.mode == "rows" and args.num > total_rows:
        raise ValueError(f"Requested {args.num} rows but train split only has {total_rows}")
    indices, lengths, skipped = sample(ds, perm, args.mode, args.num, tokenizer,
                                       args.text_field, args.seq_len)

    print(f"Saving {len(indices)} rows to {args.output}...")
    ds.to_json(indices, args.output)
    write_report(args, total_rows, configs, sizes, indices, lengths, skipped)
    print("Done!")


if __name__ == "__main__":
    main()

# python sample.py --mode tokens -n 60000000 --dataset "HiTZ/latxa-corpus-v2" --seed 42 --output latxa_corpus.jsonl --tokenizer "HiTZ/Latxa-Llama-3.1-70B-Instruct-v2" --text-field text --seq-len 8192 --report latxa_corpus-report.md
# python sample.py --mode rows -n 10000 --dataset "HiTZ/Magpie-Llama-3.1-70B-Instruct-Filtered-1M" --seed 42 --output instructions_EN.jsonl --tokenizer "HiTZ/Latxa-Llama-3.1-70B-Instruct-v2" --text-field conversations --seq-len 8192 --report instructions_EN-report.md
