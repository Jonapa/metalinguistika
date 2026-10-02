import argparse
import math

import numpy as np
from datasets import get_dataset_config_names, load_dataset, concatenate_datasets


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
                        help="Report how many sampled entries are longer than this many tokens")
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

    sizes = [len(split) for split in train_splits]
    # Concatenation of memory-mapped datasets does not copy data
    return concatenate_datasets(train_splits), configs, sizes


def to_text(value, tokenizer):
    # Chat datasets store a list of {"role", "content"} messages: render them with the
    # chat template (special tokens included) so lengths match the training format
    if isinstance(value, list):
        return tokenizer.apply_chat_template(value, tokenize=False) if value else ""
    return value or ""  # guard against null texts


def batch_lengths(ds, tokenizer, text_field, batch_idx):
    # datasets>=4 returns a lazy Column, not a list
    texts = [to_text(v, tokenizer) for v in ds.select(batch_idx)[text_field]]
    return [len(ids) for ids in tokenizer(texts, add_special_tokens=False)["input_ids"]]


def sample_rows(ds, perm, num_rows, tokenizer, text_field):
    indices = perm(np.arange(num_rows))
    lengths = []
    for start in range(0, num_rows, BATCH_SIZE):
        lengths += batch_lengths(ds, tokenizer, text_field, indices[start:start + BATCH_SIZE])
        print(f"  tokenized {len(lengths)}/{num_rows} rows")
    return indices, np.array(lengths)


def sample_tokens(ds, perm, num_tokens, tokenizer, text_field):
    selected, lengths, total_tokens = [], [], 0

    for batch_idx in perm.batches(BATCH_SIZE):
        for idx, length in zip(batch_idx, batch_lengths(ds, tokenizer, text_field, batch_idx)):
            selected.append(idx)
            lengths.append(length)
            total_tokens += length
            if total_tokens >= num_tokens:
                print(f"Reached {total_tokens} tokens with {len(selected)} rows")
                return np.array(selected), np.array(lengths)
        print(f"  {total_tokens}/{num_tokens} tokens ({len(selected)} rows)")

    print(f"Corpus exhausted: {total_tokens} tokens with {len(selected)} rows")
    return np.array(selected), np.array(lengths)


def write_report(args, total_rows, configs, sizes, indices, lengths):
    def stats(lens):
        n, over = len(lens), int((lens > args.seq_len).sum())
        pct = 100 * over / n if n else 0.0
        return n, int(lens.sum()), over, pct

    n, tokens, over, pct = stats(lengths)
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
        f"- sequence length threshold: {args.seq_len}",
        f"- output: {args.output}",
        "",
        "## Outcome",
        f"- corpus rows (train): {total_rows}",
        f"- sampled rows: {n} ({100 * n / total_rows:.4f}% of corpus)",
        f"- sampled tokens: {tokens}",
    ]
    if n:
        lines += [
            f"- tokens per entry: min {lengths.min()}, mean {lengths.mean():.1f}, "
            f"median {int(np.median(lengths))}, max {lengths.max()}",
            f"- entries longer than {args.seq_len} tokens: {over} ({pct:.2f}%)",
            f"- tokens beyond {args.seq_len} in those entries: "
            f"{int(np.clip(lengths - args.seq_len, 0, None).sum())}",
        ]

    # Map each sampled global index back to the configuration it came from
    source = np.searchsorted(np.cumsum(sizes), indices, side="right")
    lines += [
        "",
        "## Per source",
        f"| source | corpus rows | sampled rows | sampled tokens | > {args.seq_len} tokens |",
        "|---|---|---|---|---|",
    ]
    for i, (config, size) in enumerate(zip(configs, sizes)):
        n_i, tokens_i, over_i, pct_i = stats(lengths[source == i])
        lines.append(f"| {config} | {size} | {n_i} | {tokens_i} | {over_i} ({pct_i:.2f}%) |")

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

    if args.mode == "rows":
        if args.num > total_rows:
            raise ValueError(f"Requested {args.num} rows but train split only has {total_rows}")
        indices, lengths = sample_rows(ds, perm, args.num, tokenizer, args.text_field)
    else:
        indices, lengths = sample_tokens(ds, perm, args.num, tokenizer, args.text_field)

    sampled = ds.select(indices)
    print(f"Saving {len(sampled)} rows to {args.output}...")
    sampled.to_json(args.output)
    write_report(args, total_rows, configs, sizes, indices, lengths)
    print("Done!")


if __name__ == "__main__":
    main()

# python sample.py --mode tokens -n 60000000 --dataset "HiTZ/latxa-corpus-v1.1" --seed 42 --output latxa_corpus.jsonl --tokenizer "HiTZ/Latxa-Llama-3.1-70B-Instruct-v2" --text-field text --seq-len 8192 --report latxa_corpus-report.md
# python sample.py --mode rows -n 10000 --dataset "HiTZ/Magpie-Llama-3.1-70B-Instruct-Filtered-1M" --seed 42 --output instructions_EN.jsonl --tokenizer "HiTZ/Latxa-Llama-3.1-70B-Instruct-v2" --text-field conversations --seq-len 8192 --report instructions_EN-report.md