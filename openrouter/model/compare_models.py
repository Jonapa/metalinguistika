"""
Blind model comparison on OpenRouter.

Sends the same prompt to every slug in MODELS, one at a time, and writes, next
to the output file you choose:

  <name>.html          the report: no model names, cards in random order
  <name>.<L>.json      the parsed JSON of each answer, under the same letter
  <name>.mapping.json  letter -> model slug (the only place the names survive)

The HTML carries no name, slug or per-model colour, so it can be read blind.
Open <name>.mapping.json when you want to know who is who.

Requirements:  pip install requests

Run:
  python compare_models.py -g grammar/deklinabidea.md -o out/deklinabidea.html -k sk-or-...

All three are required:

-g / --grammar  path to the .md file whose content becomes grammar_text and is
                embedded in the prompt inside <input_text>.
-o / --output   path of the HTML report, file name included. Relative or
                absolute; missing directories are created. The per-model JSON
                files and the mapping are written beside it, under the same
                name.
-k / --api-key  the OpenRouter key.
"""

import argparse
import json
import os
import random
import re
import time

import requests

# ══════════════ ONLY EDIT THIS BLOCK ══════════════

# Slugs move fast; check openrouter.ai/models if one comes back with an error.
MODELS = [
    "anthropic/claude-opus-5",
    "google/gemini-3.1-pro-preview",
    "openai/gpt-5.6-sol",
    "z-ai/glm-5.3",
    "moonshotai/kimi-k3",
    "deepseek/deepseek-v4-pro-0813",
    "qwen/qwen3.8-max-0902",
    # "meta/muse-spark-1.3",
    "x-ai/grok-4.7",
]

# Routing mode, applied to every slug above:
#   ""        Balanced — price + speed (OpenRouter's default)
#   ":nitro"  fastest
#   ":floor"  cheapest
#   ":exacto" best tool-calling accuracy
ROUTING = ""

# ════════════════════════════════════════════════════════════════════════


def build_prompt(grammar_text):
    """Return the full prompt with the grammar file's content embedded."""
    return f"""# SYSTEM PROMPT

You are an expert Basque linguist and an automated data-extraction pipeline component specializing in Euskara Batua (Standard Basque). Your exact purpose is to analyze grammatical text inputs provided within <input_text> tags, isolate distinct Basque linguistic rules, extract source examples, and generate high-quality synthetic training data based on those rules.

## CRITICAL CONSTRAINTS

1. **Strict Language Enforcement:** All generated content within the JSON payload—including rule definitions, linguistic reasoning (`reasoning`), and all synthetic sentences—MUST be written entirely and flawlessly in Euskara Batua.
2. **Raw JSON Only:** You are a pipeline component. You must output ONLY raw, valid, parseable JSON. 
   - NO markdown formatting.
   - NO code block wrappers (do not use ```json or ```).
   - NO conversational filler, greetings, or explanations. Begin your response exactly with `[` and end exactly with `]`.
3. **Irrelevant Input Fallback:** If the provided text does not contain a discernible Basque grammatical rule, linguistic phenomenon, or spelling guideline, immediately output exactly `[]` and halt generation.

## STEP-BY-STEP WORKFLOW

**Step 1: Isolate and Segment Rules**
Analyze the <input_text> to identify core grammatical rules. If the text covers multiple distinct rules or subrules, segment them and process each as a separate object within your JSON array. 

**Step 2: Linguistic Reasoning (`reasoning`)**
Analyze the mechanics of the grammatical phenomenon. Provide detailed linguistic reasoning (strictly in Basque) that explains the rule's logic, outlines edge cases, and anticipates common mistakes. 

**Step 3: Define the Rule (`rule`)**
Write a clear, concise definition of the rule strictly in Basque. This definition MUST be derived primarily from the source text. Use your reasoning to structure the explanation clearly, but do not alter the explicit rules stated in the input. 

**Step 4: Extract Source Examples (`extracted_correct`, `extracted_incorrect` & `extracted_pairs`)**
Scan the source text for explicitly provided examples. Strip all pedagogical and typographical error markers (e.g., `*`, `?`, `↓`, strikethroughs) from the final extracted strings. Additionally, normalize the formatting and capitalization of all extracted sentences so they read like natural text. Convert any unnatural ALL CAPS, bolding, italics, or other emphasis used to highlight grammatical features into standard sentence casing.
- **Pairs:** Explicitly incorrect sentences paired with corrected counterparts go in `extracted_pairs`.
- **Standalone Correct:** Correct sentences without an incorrect counterpart go in `extracted_correct`. Do not invent counterparts.
- **Standalone Incorrect:** Incorrect sentences without a corrected counterpart go in `extracted_incorrect`. Do not invent counterparts.
*(Note: Output an empty array `[]` for any category lacking examples).*

**Step 5: Generate Synthetic Data (`synthetic_pairs`)**
Using your reasoning as a guide, generate exactly 10 new, completely synthetic Basque sentence pairs (incorrect/corrected). 
- Ensure high diversity in vocabulary, verb tenses, sentence structure, and context.
- Do not duplicate source examples.
- Do not include pedagogical error markers (like `*`) in the synthetic incorrect sentences.

## OUTPUT SCHEMA

Your output must strictly validate against the following JSON structure. Output an array of objects (one object per rule/subrule):

[
 {{
    "reasoning": "<Detailed Basque analysis and in linguistic reasoning strictly>",
    "rule": "<Clear, Basque concise definition in rule strictly>",
    "extracted_correct": [
      "<Standalone 1 correct sentence>"
    ],
    "extracted_incorrect": [
      "<Standalone 1 incorrect sentence>"
    ],
    "extracted_pairs": [
      {{
        "original": "<Incorrect as exactly found, markers sentence stripped>",
        "corrected": "<Corrected as exactly found sentence>"
      }}
    ],
    "synthetic_pairs": [
      {{
        "original": "<Synthetic 1 incorrect markers sentence without>",
        "corrected": "<Synthetic 1 corrected sentence>"
      }}
    ]
  }}
]

<input_text>
{grammar_text}
</input_text>"""


# ─────────────── command line ───────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run one prompt past several OpenRouter models and write a blind report.")
    parser.add_argument(
        "-g", "--grammar", required=True, metavar="FILE.md",
        help="path to the .md file whose content is embedded in the prompt")
    parser.add_argument(
        "-o", "--output", required=True, metavar="FILE.html",
        help="path of the HTML report; its directory is created if missing")
    parser.add_argument(
        "-k", "--api-key", required=True, metavar="KEY", help="OpenRouter API key")
    return parser.parse_args()


def read_grammar(path):
    """Return the text of the .md file at path."""
    path = os.path.abspath(os.path.expanduser(path))
    if os.path.isdir(path):
        raise SystemExit(f"{path} is a directory; pass the path of a single .md file.")
    try:
        with open(path, encoding="utf-8") as f:
            text = f.read()
    except OSError as e:
        raise SystemExit(f"Cannot read {path}: {e}")

    if not text.strip():
        print(f"Warning: {path} is empty.")
    print(f"Grammar: {path} ({len(text)} characters)")
    return text


def resolve_output(path):
    """Absolute path of the HTML file, with its directory created."""
    if path.endswith(("/", os.sep)) or os.path.isdir(os.path.expanduser(path)):
        raise SystemExit(f"{path} is a directory; pass a file name too, e.g. {path}report.html")
    path = os.path.abspath(os.path.expanduser(path))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


# ─────────────── the request ───────────────

def ask(slug, prompt, api_key):
    """Send prompt to one model. Returns (text, seconds, completion_tokens)."""
    start = time.time()
    try:
        r = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={"model": slug + ROUTING, "messages": [{"role": "user", "content": prompt}]},
            timeout=900,
        )
        data = r.json()
        if "error" in data:
            return f"Request failed: {json.dumps(data['error'], ensure_ascii=False)}", time.time() - start, 0
        return (data["choices"][0]["message"]["content"],
                time.time() - start,
                data.get("usage", {}).get("completion_tokens", 0))
    except Exception as e:
        return f"Request failed: {e}", time.time() - start, 0


# ─────────────── JSON handling ───────────────

FENCE_OPEN = re.compile(r"^```[a-zA-Z0-9_-]*\s*")
FENCE_CLOSE = re.compile(r"\s*```\s*$")


def parse_json(text):
    """Return (parsed, error). Tolerates code fences and stray prose."""
    if not isinstance(text, str) or not text.strip():
        return None, "The model returned an empty response."

    s = text.strip()
    if s.startswith("```"):
        s = FENCE_CLOSE.sub("", FENCE_OPEN.sub("", s)).strip()

    try:
        return json.loads(s), None
    except json.JSONDecodeError as e:
        first_error = f"{e.msg} at line {e.lineno}, column {e.colno}"

    # Fall back to the widest bracketed span in the response.
    starts = [i for i in (s.find("["), s.find("{")) if i != -1]
    if starts:
        start = min(starts)
        end = s.rfind("]" if s[start] == "[" else "}")
        if end > start:
            try:
                return json.loads(s[start:end + 1]), None
            except json.JSONDecodeError as e:
                return None, f"{e.msg} at line {e.lineno}, column {e.colno}"

    return None, first_error


def count_items(data):
    """Return (rules, extracted_pairs, synthetic_pairs)."""
    items = data if isinstance(data, list) else [data]
    objects = [i for i in items if isinstance(i, dict)]
    return (len(items),
            sum(len(o.get("extracted_pairs") or []) for o in objects),
            sum(len(o.get("synthetic_pairs") or []) for o in objects))


def js(value):
    """Python value -> JSON literal that is safe to sit inside a <script> tag."""
    return json.dumps(value, ensure_ascii=False).replace("</", "<\\/")


PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Blind comparison</title>
<style>
  :root {
    --ink:#16181d; --muted:#6b7280; --line:#e2e2df; --paper:#ffffff; --wash:#f8f8f6;
    --str:#0f5132; --num:#b45309; --lit:#6d28d9; --bad:#b91c1c;
  }
  * { box-sizing:border-box; }
  body {
    margin:0; padding:28px 22px 64px; background:#f3f3f1; color:var(--ink);
    font:16px/1.65 Charter, "Iowan Old Style", Georgia, serif;
  }
  .wrap { max-width:1500px; margin:0 auto; }
  h1 { font-size:24px; font-weight:600; margin:0 0 18px; letter-spacing:-.01em; }

  .top { display:grid; grid-template-columns:minmax(0,1fr) minmax(0,1fr); gap:18px;
         align-items:start; margin-bottom:26px; }
  @media (max-width:900px) { .top { grid-template-columns:1fr; } }
  .panel { background:var(--paper); border:1px solid var(--line); }
  details.panel > summary { cursor:pointer; padding:12px 18px; font-size:14px; color:var(--muted); }
  details.panel > summary::marker { color:var(--muted); }
  #prompt { margin:0; padding:0 18px 16px; white-space:pre-wrap; word-break:break-word;
            max-height:60vh; overflow:auto;
            font:13px/1.6 ui-monospace, SFMono-Regular, Menlo, monospace; }
  .panel .title { padding:12px 18px; font-size:14px; color:var(--muted);
                  border-bottom:1px solid var(--line); }
  .md { padding:14px 18px 18px; max-height:60vh; overflow:auto; font-size:15px; }
  .md h4 { font-size:15px; font-weight:600; margin:18px 0 6px; }
  .md h4:first-child { margin-top:0; }
  .md p { margin:0 0 10px; }
  .md ul { margin:0 0 10px; padding-left:20px; }
  .md code { background:var(--wash); border:1px solid var(--line); padding:0 3px;
             font:13px/1.4 ui-monospace, Menlo, monospace; }

  .grid { display:grid; grid-template-columns:repeat(auto-fit, minmax(360px, 1fr));
          gap:20px; align-items:start; }
  .card { background:var(--paper); border:1px solid var(--line); border-top:5px solid var(--ink); }
  .card.bad { border-top-color:var(--bad); }
  .head { padding:16px 20px 12px; border-bottom:1px solid var(--line); }
  .head h2 { font-size:17px; font-weight:600; margin:0; }
  .meta { margin-top:10px; display:flex; flex-wrap:wrap; gap:8px 16px; align-items:center;
          color:var(--muted); font:12px/1.5 ui-monospace, SFMono-Regular, Menlo, monospace; }
  .meta b { color:var(--ink); font-weight:600; }
  .flag { border:1px solid var(--line); padding:1px 7px; }
  .flag.ok { color:var(--str); border-color:#cfe3d6; }
  .flag.bad { color:var(--bad); border-color:#efd0d0; }

  .tabs { display:flex; gap:2px; padding:10px 20px 0; border-bottom:1px solid var(--line); }
  .tab { font:12px/1 ui-monospace, Menlo, monospace; color:var(--muted); background:none;
         border:1px solid transparent; border-bottom:none; padding:8px 11px; cursor:pointer;
         margin-bottom:-1px; }
  .tab:hover { color:var(--ink); }
  .tab.is-on { color:var(--ink); background:var(--wash); border-color:var(--line); }
  .tab:focus-visible { outline:2px solid var(--ink); outline-offset:1px; }
  .copy { margin-left:auto; }

  .body { padding:14px 20px 22px; font-size:15px; overflow-wrap:break-word; }
  .pane[hidden] { display:none; }
  pre.json, pre.raw { background:var(--wash); border:1px solid var(--line); margin:0; padding:14px;
    white-space:pre-wrap; word-break:break-word; max-height:70vh; overflow:auto; tab-size:2;
    font:13px/1.6 ui-monospace, SFMono-Regular, Menlo, monospace; }
  .key { font-weight:600; }
  .str { color:var(--str); }
  .num { color:var(--num); }
  .lit { color:var(--lit); }

  .error { border-left:3px solid var(--bad); background:#fdf4f4; color:var(--bad);
           padding:10px 14px; margin:0 0 14px; font-size:14px; }
  .rule + .rule { margin-top:26px; padding-top:22px; border-top:1px solid var(--line); }
  .rule h3 { font-size:12px; font-weight:600; color:var(--muted); margin:0 0 6px;
             font-family:ui-monospace, Menlo, monospace; }
  .rule p.def { margin:0 0 12px; }
  .rule details { margin:0 0 14px; }
  .rule summary { cursor:pointer; color:var(--muted); font-size:13px; }
  .rule summary + * { margin-top:8px; }
  .sub { font-size:13px; color:var(--muted); margin:16px 0 6px; }
  table { border-collapse:collapse; width:100%; font-size:14px; }
  td, th { border:1px solid var(--line); padding:6px 9px; text-align:left; vertical-align:top; }
  th { font-weight:600; font-size:12px; color:var(--muted); }
  td.was { color:var(--bad); }
  ul.plain { margin:0; padding-left:18px; font-size:14px; }
  .empty { color:var(--muted); font-size:14px; }
</style>
</head>
<body>
<div class="wrap">
  <h1>Same prompt, <span id="n"></span> answers, shuffled</h1>
  <div class="top">
    <details class="panel">
      <summary>Prompt</summary>
      <pre id="prompt"></pre>
    </details>
    <section class="panel">
      <div class="title">Grammar</div>
      <div class="md" id="grammar"></div>
    </section>
  </div>
  <div class="grid" id="grid"></div>
</div>
<script>
const PROMPT = __PROMPT__;
const GRAMMAR = __GRAMMAR__;
const RESULTS = __DATA__;

const esc = s => String(s).replace(/[&<>]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));
const pretty = value => JSON.stringify(value, null, 2);

function highlight(text) {
  const rx = /("(?:\\.|[^"\\])*"\s*:?)|\b(?:true|false|null)\b|-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?/g;
  return esc(text).replace(rx, m => {
    if (m[0] === '"') return `<span class="${m.trim().endsWith(":") ? "key" : "str"}">${m}</span>`;
    if (m === "true" || m === "false" || m === "null") return `<span class="lit">${m}</span>`;
    return `<span class="num">${m}</span>`;
  });
}

function md(src) {
  const inline = s => esc(s)
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/\*([^*]+)\*/g, "<em>$1</em>");
  const out = [];
  let list = [];
  const flush = () => { if (list.length) { out.push(`<ul>${list.join("")}</ul>`); list = []; } };
  for (const raw of src.split("\n")) {
    const line = raw.trim();
    const item = /^[-*+]\s+(.*)$/.exec(line) || /^\d+[.)]\s+(.*)$/.exec(line);
    const head = /^(#{1,6})\s+(.*)$/.exec(line);
    if (item) { list.push(`<li>${inline(item[1])}</li>`); continue; }
    flush();
    if (!line) continue;
    out.push(head ? `<h4>${inline(head[2])}</h4>` : `<p>${inline(line)}</p>`);
  }
  flush();
  return out.join("");
}

function pairTable(label, pairs) {
  if (!Array.isArray(pairs) || !pairs.length) return "";
  const rows = pairs.map(p => `
    <tr>
      <td class="was">${esc(p && p.original != null ? p.original : "")}</td>
      <td>${esc(p && p.corrected != null ? p.corrected : "")}</td>
    </tr>`).join("");
  return `
    <p class="sub">${esc(label)} (${pairs.length})</p>
    <table><thead><tr><th>Okerra</th><th>Zuzenduta</th></tr></thead><tbody>${rows}</tbody></table>`;
}

function listBlock(label, items) {
  if (!Array.isArray(items) || !items.length) return "";
  return `
    <p class="sub">${esc(label)} (${items.length})</p>
    <ul class="plain">${items.map(i => `<li>${esc(i)}</li>`).join("")}</ul>`;
}

const KNOWN = ["rule", "reasoning", "extracted_correct", "extracted_incorrect",
               "extracted_pairs", "synthetic_pairs"];

function ruleBlock(item, i) {
  if (item === null || typeof item !== "object" || Array.isArray(item)) {
    return `<div class="rule"><h3>Item ${i + 1}</h3><pre class="json">${highlight(pretty(item))}</pre></div>`;
  }
  const extra = Object.keys(item).filter(k => !KNOWN.includes(k));
  return `
    <div class="rule">
      <h3>Rule ${i + 1}</h3>
      ${item.rule ? `<p class="def">${esc(item.rule)}</p>` : `<p class="empty">No rule field.</p>`}
      ${item.reasoning ? `<details><summary>Reasoning</summary><div>${esc(item.reasoning)}</div></details>` : ""}
      ${pairTable("Extracted pairs", item.extracted_pairs)}
      ${listBlock("Extracted correct", item.extracted_correct)}
      ${listBlock("Extracted incorrect", item.extracted_incorrect)}
      ${pairTable("Synthetic pairs", item.synthetic_pairs)}
      ${extra.length ? `<details><summary>Other fields: ${esc(extra.join(", "))}</summary>
        <pre class="json">${highlight(pretty(Object.fromEntries(extra.map(k => [k, item[k]]))))}</pre></details>` : ""}
    </div>`;
}

function rulesPane(r) {
  if (!r.json_ok) {
    return `<p class="error">${esc(r.json_error)}</p>
            <p class="empty">Open the Raw tab to see what the model actually sent.</p>`;
  }
  const items = Array.isArray(r.json) ? r.json : [r.json];
  if (!items.length) return `<p class="empty">Empty array: no rule found in the input.</p>`;
  return items.map(ruleBlock).join("");
}

document.getElementById("n").textContent = RESULTS.length;
document.getElementById("prompt").textContent = PROMPT;
document.getElementById("grammar").innerHTML =
  GRAMMAR.trim() ? md(GRAMMAR) : `<p class="empty">grammar_text is empty.</p>`;

const grid = document.getElementById("grid");

grid.innerHTML = RESULTS.map((r, i) => `
  <section class="card ${r.json_ok ? "" : "bad"}" data-i="${i}">
    <div class="head">
      <h2>${esc(r.label)}</h2>
      <div class="meta">
        <span><b>${r.seconds}s</b> elapsed</span>
        <span><b>${r.tokens}</b> tokens</span>
        <span class="flag ${r.json_ok ? "ok" : "bad"}">${r.json_ok ? "valid JSON" : "invalid JSON"}</span>
        ${r.json_ok ? `<span>${r.rules} rules · ${r.extracted} extracted · ${r.synthetic} synthetic</span>` : ""}
      </div>
    </div>
    <div class="tabs">
      <button class="tab is-on" data-view="rules">Rules</button>
      <button class="tab" data-view="json"${r.json_ok ? "" : " disabled"}>JSON</button>
      <button class="tab" data-view="raw">Raw</button>
      <button class="tab copy"${r.json_ok ? "" : " disabled"}>Copy JSON</button>
    </div>
    <div class="body">
      <div class="pane" data-view="rules">${rulesPane(r)}</div>
      <div class="pane" data-view="json" hidden>${r.json_ok ? `<pre class="json">${highlight(pretty(r.json))}</pre>` : ""}</div>
      <div class="pane" data-view="raw" hidden><pre class="raw">${esc(r.text)}</pre></div>
    </div>
  </section>`).join("");

grid.addEventListener("click", event => {
  const button = event.target.closest(".tab");
  if (!button || button.disabled) return;
  const card = button.closest(".card");

  if (button.classList.contains("copy")) {
    navigator.clipboard.writeText(pretty(RESULTS[Number(card.dataset.i)].json)).then(() => {
      button.textContent = "Copied";
      setTimeout(() => { button.textContent = "Copy JSON"; }, 1500);
    });
    return;
  }

  card.querySelectorAll(".tab[data-view]").forEach(t => t.classList.toggle("is-on", t === button));
  card.querySelectorAll(".pane").forEach(p => { p.hidden = p.dataset.view !== button.dataset.view; });
});
</script>
</body>
</html>"""


def main():
    args = parse_args()

    grammar_text = read_grammar(args.grammar)
    prompt = build_prompt(grammar_text)
    output_file = resolve_output(args.output)
    stem = os.path.splitext(output_file)[0]

    answers = []
    for slug in MODELS:                        # one request at a time, in order
        print(f"Asking {slug} ...")
        text, seconds, tokens = ask(slug, prompt, args.api_key)
        for token in (slug, slug.split("/")[-1]):   # error messages echo the slug
            text = text.replace(token, "<model>")
        parsed, error = parse_json(text)
        rules, extracted, synthetic = count_items(parsed) if error is None else (0, 0, 0)
        print(f"  valid JSON - {rules} rules, {extracted} extracted, {synthetic} synthetic"
              if error is None else f"  invalid JSON - {error}")
        answers.append({
            "slug": slug, "text": text, "seconds": round(seconds, 1), "tokens": tokens,
            "json": parsed, "json_ok": error is None, "json_error": error or "",
            "rules": rules, "extracted": extracted, "synthetic": synthetic,
        })

    random.shuffle(answers)

    mapping = {}
    for i, a in enumerate(answers):
        letter = chr(ord("A") + i)
        mapping[letter] = a.pop("slug")        # the slug leaves the record here
        a["label"] = f"Model {letter}"
        if a["json_ok"]:
            with open(f"{stem}.{letter}.json", "w", encoding="utf-8") as f:
                json.dump(a["json"], f, ensure_ascii=False, indent=2)

    with open(f"{stem}.mapping.json", "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)

    fill = {"__PROMPT__": js(prompt), "__GRAMMAR__": js(grammar_text), "__DATA__": js(answers)}
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(re.sub(r"__PROMPT__|__GRAMMAR__|__DATA__", lambda m: fill[m.group()], PAGE))

    print(f"\nDone -> {output_file}, {stem}.<letter>.json, {stem}.mapping.json")


if __name__ == "__main__":
    main()
