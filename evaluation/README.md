# 1.- Save evaluation dataset
    - Save `metalinguistika_galderak` as .csv inside input/
# 2.- Prepare evaluation dataset for inference
    -  <output>.template.json: contains the header (first row)

```bash
#!/usr/bin/env bash

DATASET="08-11-26"

python src/00_preprocess_infer.py \
    --path "input/${DATASET}/${DATASET}.csv" \
    --output-path "input/${DATASET}/${DATASET}.preprocessed.json"
```

# 3.- Do inference
    - Copies the input JSON and adds to each row an `infer` field with `infer_model`, `infer_user_message` and `infer_response`.
    - If the output file already exists, only rows without an `infer.infer_response` are queried (rerun to retry failures).

```bash
#!/usr/bin/env bash

DATASET="08-11-26"
FILENAME="${DATASET}.preprocessed"
API_KEY="${OPENROUTER_API_KEY:-sk-or-xxxx}"   # env var takes precedence

# OpenRouter model names to evaluate
INFER_MODELS=(
  "google/gemini-3.1-pro-preview"
  "openai/gpt-6.1-sol"
  "anthropic/claude-sonnet-5.5"
)

for INFER_MODEL in "${INFER_MODELS[@]}"; do
  python src/01_infer.py \
      --input-path "input/${DATASET}/${FILENAME}.json" \
      --output-path "output/infer/${FILENAME}__${INFER_MODEL//[\/:]/_}.json" \
      --model "$INFER_MODEL" \
      --api-key "$API_KEY"
done
```

    - For Latxa (self-hosted vLLM server, OpenAI-compatible API) use `01_infer_latxa.py` instead: same input/output format, it adds Latxa's system prompt and chat template. Model, server URL and API key are fixed in the script; it waits `SLEEP_BETWEEN_QUERIES` seconds between queries to avoid overloading the server.

```bash
#!/usr/bin/env bash

DATASET="08-11-26"
FILENAME="${DATASET}.preprocessed"
INFER_MODEL="Latxa-Llama-3.1-70B-Instruct-exp_2_101_v2-FP8"

python src/01_infer_latxa.py \
    --input-path "input/${DATASET}/${FILENAME}.json" \
    --output-path "output/infer/${FILENAME}__${INFER_MODEL//[\/:]/_}.json"
```

# 4.- Prepare inference output for the judge
    - Adds to each row a `judge` field with `judge_system_message` and `judge_user_message`, the latter built from `input/judge/correct_judge_user.txt` (answer 1) or `input/judge/incorrect_judge_user.txt` (answer 0).
    - Rows without an `infer.infer_response` (failed inference) are skipped: rerun step 3 first.

```bash
#!/usr/bin/env bash

DATASET="08-11-26"
FILENAME="${DATASET}.preprocessed"

# OpenRouter model names to evaluate
INFER_MODELS=(
    "google/gemini-3.1-pro-preview"
    "openai/gpt-6.1-sol"
    "anthropic/claude-sonnet-5.5"
    "Latxa-Llama-3.1-70B-Instruct-exp_2_101_v2-FP8"
)

for INFER_MODEL in "${INFER_MODELS[@]}"; do
    python src/02_preprocess_judge.py \
        --input-path "output/infer/${FILENAME}__${INFER_MODEL//[\/:]/_}.json" \
        --output-path "output/judge/${FILENAME}__${INFER_MODEL//[\/:]/_}.json" \
        --system-prompt input/judge_prompts/judge_system.txt \
        --correct-prompt input/judge_prompts/correct_judge_user.txt \
        --incorrect-prompt input/judge_prompts/incorrect_judge_user.txt
done
```

# 5.- Do judge evaluation
    - Copies the judge JSON, sends `judge_system_message` + `judge_user_message` to the judge model and adds to each row a `result` field with the parsed JSON (`arrazoiketa` and `baldintzak`). The judge model is kept in `judge.judge_model`.
    - Output file name: `<filename>__<infer model>__<judge model>.json`.
    - If the output file already exists, only rows without a `result` are queried (rerun to retry failures or unparseable answers).

```bash
#!/usr/bin/env bash

DATASET="08-11-26"
FILENAME="${DATASET}.preprocessed"

# OpenRouter model names to evaluate
INFER_MODELS=(
    "google/gemini-3.1-pro-preview"
    "openai/gpt-6.1-sol"
    "anthropic/claude-sonnet-5.5"
    "Latxa-Llama-3.1-70B-Instruct-exp_2_101_v2-FP8"
)

# OpenRouter model names to evaluate
JUDGE_MODELS=(
    "google/gemini-3.1-pro-preview"
    "openai/gpt-6.1-sol"
    "anthropic/claude-sonnet-5.5"
)

API_KEY="${OPENROUTER_API_KEY:-sk-or-xxxx}"   # env var takes precedence

for JUDGE_MODEL in "${JUDGE_MODELS[@]}"; do
    for INFER_MODEL in "${INFER_MODELS[@]}"; do
        python src/03_judge.py \
            --input-path "output/judge/${FILENAME}__${INFER_MODEL//[\/:]/_}.json" \
            --output-path "output/results/${FILENAME}__${INFER_MODEL//[\/:]/_}__${JUDGE_MODEL//[\/:]/_}.json" \
            --model "$JUDGE_MODEL" \
            --api-key "$API_KEY"
    done
done
```

    - For Latxa (self-hosted vLLM server, OpenAI-compatible API) use `03_judge_latxa.py` instead.

```bash
#!/usr/bin/env bash

DATASET="08-11-26"
FILENAME="${DATASET}.preprocessed"

# OpenRouter model names to evaluate
INFER_MODELS=(
    "google/gemini-3.1-pro-preview"
    "openai/gpt-6.1-sol"
    "anthropic/claude-sonnet-5.5"
    "Latxa-Llama-3.1-70B-Instruct-exp_2_101_v2-FP8"
)

JUDGE_MODEL="Latxa-Llama-3.1-70B-Instruct-exp_2_101_v2-FP8"

for INFER_MODEL in "${INFER_MODELS[@]}"; do
    python src/03_judge_latxa.py \
        --input-path "output/judge/${FILENAME}__${INFER_MODEL//[\/:]/_}.json" \
        --output-path "output/results/${FILENAME}__${INFER_MODEL//[\/:]/_}__${JUDGE_MODEL//[\/:]/_}.json" 
done
```
