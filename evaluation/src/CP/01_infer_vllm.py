import argparse
import importlib.util
import json
import sys
from pathlib import Path

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

# Reuse Latxa's chat template and system prompt so the setting is the same as 01_infer_latxa.py
_spec = importlib.util.spec_from_file_location("infer_latxa", Path(__file__).parent / "01_infer_latxa.py")
infer_latxa = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(infer_latxa)
CUSTOM_CHAT_TEMPLATE = infer_latxa.CUSTOM_CHAT_TEMPLATE
SYSTEM_PROMPT_DEFAULT = infer_latxa.SYSTEM_PROMPT_DEFAULT

SAMPLING_PARAMS = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=32768)

# Defaults for Latxa 70B in bf16 (~141 GB of weights) on one full MN5 ACC node (4x H100 64 GB)
TENSOR_PARALLEL_SIZE = 4  # must divide the 64 attention / 8 KV heads of Llama-3.1-70B
MAX_MODEL_LEN = 40960  # 32768 generated tokens + system prompt + question


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--model", required=True, help="Local checkpoint path or HF model id")
    parser.add_argument("--model-name", help="Name stored in infer_model (default: basename of --model)")
    parser.add_argument("--tensor-parallel-size", type=int, default=TENSOR_PARALLEL_SIZE)
    parser.add_argument("--max-model-len", type=int, default=MAX_MODEL_LEN)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    args = parser.parse_args()
    model_name = args.model_name or Path(args.model.rstrip("/")).name

    # Resume from the output if it exists: rows without a response are (re)queried
    path = args.output_path if args.output_path.exists() else args.input_path
    rows = json.loads(path.read_text(encoding="utf-8"))
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    todo = [row for row in rows if not row.get("infer", {}).get("infer_response")]

    # Render the prompt ourselves (template already adds bos_token) so BOS is not duplicated at tokenization
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    prompts = []
    for row in todo:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_DEFAULT},
            {"role": "user", "content": row["question"]},
        ]
        text = tokenizer.apply_chat_template(
            messages, chat_template=CUSTOM_CHAT_TEMPLATE, add_generation_prompt=True, tokenize=False
        )
        prompts.append(TokensPrompt(prompt_token_ids=tokenizer.encode(text, add_special_tokens=False)))

    # Offline vLLM only checks the prompt length and would silently cut generations at max_model_len
    longest = max((len(p["prompt_token_ids"]) for p in prompts), default=0)
    if longest + SAMPLING_PARAMS.max_tokens > args.max_model_len:
        sys.exit(f"--max-model-len {args.max_model_len} < longest prompt ({longest}) + max_tokens ({SAMPLING_PARAMS.max_tokens})")

    n_errors = 0
    if prompts:
        llm = LLM(
            model=args.model,
            dtype="bfloat16",
            tensor_parallel_size=args.tensor_parallel_size,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
        )
        outputs = llm.generate(prompts, SAMPLING_PARAMS)
        for row, output in zip(todo, outputs):
            response = output.outputs[0].text
            if not response.strip():
                n_errors += 1
                continue
            row["infer"] = {"infer_model": model_name, "infer_user_message": row["question"], "infer_response": response}
        args.output_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"{args.output_path}: {n_errors} errors (rerun to retry them)")
    sys.exit(1 if n_errors else 0)
