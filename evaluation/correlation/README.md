# Correlation

Compare LLM-judge annotations against the human gold standard.

## 0. Group by judge

One results file per judge.

```bash
python3 0_group_by_judge.py -i results/ -o results/emaitzak/
```

## 1. Human annotations to results

Convert human annotations (`anotazioak_*.json`) to the results format (gold).

```bash
python3 1_anotazioak_to_results.py -i anotazioak_2026-10-18_1635.json -o results/emaitzak/
```

## 2. Judge vs gold correlation

Leaderboard (Cohen's κ with 95% CI, raw agreement, F1, item-score Spearman) in one figure, one row each for incorrect (5 baldintzak), correct (2 baldintzak) and both pooled.

```bash
python3 2_judgeOfJudge_gold_correlation.py \
    --gold results/emaitzak/human_anotazioak_2026-10-18_1635.json \
    --judges results/emaitzak/anthropic_claude-sonnet-5.5.json \
             results/emaitzak/google_gemini-3.1-pro-preview.json \
             results/emaitzak/Latxa-Llama-3.1-70B-Instruct-exp_2_101_v2-FP8.json \
             results/emaitzak/openai_gpt-6.1-sol.json \
             results/emaitzak/google_gemini-4-argon.json \
    --out results/emaitzak/correlation_results/
```

## 3. Baldintza pass rates

Gold pass rate per baldintza (incorrect and correct sentences), and the cumulative share meeting B1, B1 & B5, all five (incorrect) / B1, both (correct). Each graph adds an "all sentences" panel pooling B1 and B5 (incorrect) / B2 (correct). `--gold` is the output of step 1, not the raw `anotazioak_*.json`.

```bash
python3 3_judge_baldintza.py \
    --gold results/emaitzak/human_anotazioak_2026-10-18_1635.json \
    --out results/emaitzak/baldintza_results/
```
