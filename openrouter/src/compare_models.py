"""
Run one prompt past several OpenRouter models in parallel and save each answer
as JSON.

Every slug in MODELS gets its own thread, which works through the grammar files
one after another. An answer is retried (up to MAX_ATTEMPTS) until it is valid
JSON with the expected schema, then written to

  <output>/<grammar name>/<slug>.json         the answer
  <output>/<grammar name>/<slug>.meta.json    served model version, generation
                                              id, finish reason, provider,
                                              request settings, the model's
                                              default hyperparameters and
                                              endpoint details as listed by
                                              OpenRouter, usage and cost
                                              (last attempt and all attempts),
                                              time, attempts, reasoning trace,
                                              request hash, and the inputs: the
                                              full prompt sent, the prompt and
                                              grammar file paths and the
                                              grammar text
  <output>/<grammar name>/<slug>.failed.txt   only when every attempt failed:
                                              the last error and raw answer

with "/" -> "__" in the slug. An answer is skipped when it already exists and
was made from the same request (prompt template + grammar text + REASONING +
the model's providers), so a run can be resumed after a crash or Ctrl-C; change
any of them and it is redone.

Errors that will not go away (bad slug, bad key, no credits) stop that model;
timeouts, rate limits, 5xx and invalid JSON are retried.

Requirements:  pip install requests

Run:
  python src/compare_models.py -p src/prompt.md -g input/jagonet_0.md input/jagonet_1.md -o output -k sk-or-...
  python src/compare_models.py -p src/prompt.md -g input/ -o output -k sk-or-...

-p / --prompt   .md file with the prompt template; "{grammar_text}" is replaced
                by the grammar file's content, "{{" and "}}" become "{" and "}".
-g / --grammar  one or more .md files, or directories of .md files, whose
                content becomes grammar_text.
-o / --output   directory for the results; created if missing.
-k / --api-key  the OpenRouter key.
"""

import argparse
import glob
import hashlib
import json
import os
import re
import threading
import time

import requests

# ══════════════ ONLY EDIT THIS BLOCK ══════════════

# Model slug -> providers it may be served from (OpenRouter provider slugs).
# An empty list lets OpenRouter choose.
# Slugs move fast; check openrouter.ai/models if one keeps coming back with an error.
MODELS = {
    "deepseek/deepseek-v4.1-flash-1017": ["together"],
    "deepseek/deepseek-v4-pro-0813":     ["together"],
    "z-ai/glm-5.3":                      ["z-ai/fp8"],
    "moonshotai/kimi-k3":                ["moonshotai/mxfp4"],
    "qwen/qwen3.8-max-0902":             ["alibaba"],
    "meta/muse-spark-open":              ["together"],

    "openai/gpt-6.1-sol":                ["openai"],
    "anthropic/claude-sonnet-5.5":       ["anthropic"],
    "google/gemini-4-argon":             ["google-vertex/global"],
    "google/gemini-3.1-pro-preview":     ["google-vertex/global"],
}

REASONING = {"effort": "high"}

# Seconds to wait between retries of the same model (doubles up to RETRY_MAX).
RETRY_WAIT = 5
RETRY_MAX = 60
# Attempts per grammar file before giving up on it.
MAX_ATTEMPTS = 10

# ════════════════════════════════════════════════════════════════════════

FENCE = re.compile(r"^```[a-zA-Z0-9_-]*\s*|\s*```\s*$")

# HTTP codes that retrying will not fix: bad request / unknown model, bad key,
# no credits, forbidden, not found.
PERMANENT = {400, 401, 402, 403, 404}

STRING_LISTS = ("extracted_correct", "extracted_incorrect", "synthetic_correct")
PAIR_LISTS = ("extracted_pairs", "synthetic_pairs")

stop = threading.Event()


class PermanentError(Exception):
    """An error that retrying will not fix; stops the model."""


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run one prompt past several OpenRouter models in parallel and save the JSON answers.")
    parser.add_argument("-p", "--prompt", required=True, metavar="FILE.md",
                        help="prompt template; {grammar_text} is replaced by the grammar file")
    parser.add_argument("-g", "--grammar", required=True, nargs="+", metavar="FILE.md|DIR",
                        help=".md files or directories of .md files to embed in the prompt")
    parser.add_argument("-o", "--output", required=True, metavar="DIR",
                        help="directory for the results")
    parser.add_argument("-k", "--api-key", required=True, metavar="KEY",
                        help="OpenRouter API key")
    return parser.parse_args()


def read(path):
    with open(os.path.expanduser(path), encoding="utf-8") as f:
        return f.read()


def grammar_files(paths):
    """Expand directories into their .md files, keeping the order given."""
    files = []
    for p in paths:
        p = os.path.expanduser(p)
        files += sorted(glob.glob(os.path.join(p, "*.md"))) if os.path.isdir(p) else [p]
    if not files:
        raise SystemExit("No .md grammar files found.")
    return files


def build_prompt(template, grammar_text):
    """Unescape {{ }} in the template, then embed the grammar text."""
    template = template.replace("{{", "{").replace("}}", "}")
    return template.replace("{grammar_text}", grammar_text)


def ask(slug, prompt, api_key):
    """Send prompt to one model and return the response JSON; raises on failure."""
    body = {
        "model": slug,
        "messages": [{"role": "user", "content": prompt}],
        "reasoning": REASONING,
    }
    if MODELS[slug]:
        body["provider"] = {"only": MODELS[slug]}
    r = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json=body,
        timeout=900,
    )
    try:
        data = r.json()
    except ValueError:
        data = None

    # OpenRouter may also put an error in a 200 body, with its own code.
    error = data.get("error") if isinstance(data, dict) else None
    code = r.status_code if r.status_code != 200 else (error or {}).get("code", 200)
    if r.status_code != 200 or error:
        detail = json.dumps(error, ensure_ascii=False) if error else r.text[:200].strip()
        message = f"HTTP {code}: {detail}"
        raise PermanentError(message) if code in PERMANENT else RuntimeError(message)
    if data is None:
        raise RuntimeError(f"response is not JSON: {r.text[:200].strip()}")
    return data


def fetch_catalog():
    """OpenRouter's public model list, by slug; empty if it cannot be fetched.
    Read only: nothing in it is sent with the requests."""
    try:
        r = requests.get("https://openrouter.ai/api/v1/models", timeout=60)
        r.raise_for_status()
        return {m["id"]: m for m in r.json()["data"]}
    except Exception as e:
        print(f"could not fetch the model list, meta will lack defaults: {e}", flush=True)
        return {}


def model_info(slug, catalog):
    """What OpenRouter lists about the model: its default hyperparameters,
    reasoning defaults and the endpoints it may be served from."""
    model = catalog.get(slug, {})
    info = {
        "canonical_slug": model.get("canonical_slug"),
        "default_parameters": model.get("default_parameters"),
        "reasoning": model.get("reasoning"),
        "context_length": model.get("context_length"),
    }
    try:
        r = requests.get(f"https://openrouter.ai/api/v1/models/{slug}/endpoints", timeout=60)
        r.raise_for_status()
        endpoints = r.json()["data"]["endpoints"]
    except Exception as e:
        info["endpoints_error"] = str(e)
        return info
    # Keep only the endpoints MODELS allows ("deepseek" also covers "deepseek/fp8").
    allowed = MODELS[slug]
    info["endpoints"] = [
        {key: ep.get(key) for key in ("provider_name", "tag", "quantization", "context_length",
                                      "max_completion_tokens", "supported_parameters", "pricing")}
        for ep in endpoints
        if not allowed or any(ep.get("tag") == p or (ep.get("tag") or "").startswith(p + "/")
                              for p in allowed)
    ]
    return info


def parse_json(text):
    """json.loads after stripping a ```json fence; raises if invalid."""
    if not text or not text.strip():
        raise ValueError("empty response")
    return json.loads(FENCE.sub("", text.strip()))


def check_schema(data):
    """Raise ValueError unless data is a list of rule objects as the prompt asks."""
    if not isinstance(data, list):
        raise ValueError(f"expected a list, got {type(data).__name__}")
    for n, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"item {n} is not an object")
        for key in ("reasoning", "rule"):
            if not isinstance(item.get(key), str):
                raise ValueError(f"item {n}: '{key}' missing or not a string")
        for key in STRING_LISTS:
            value = item.get(key)
            if not isinstance(value, list) or not all(isinstance(s, str) for s in value):
                raise ValueError(f"item {n}: '{key}' missing or not a list of strings")
        for key in PAIR_LISTS:
            value = item.get(key)
            if not isinstance(value, list) or not all(
                    isinstance(p, dict) and isinstance(p.get("original"), str)
                    and isinstance(p.get("corrected"), str) for p in value):
                raise ValueError(f"item {n}: '{key}' missing or not a list of original/corrected pairs")


def write_atomic(path, text):
    """Write to a temp file and rename it into place, so a kill mid-write never
    leaves a half-written file behind."""
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, path)


def write_json(path, data):
    write_atomic(path, json.dumps(data, ensure_ascii=False, indent=2))


def add_usage(total, usage):
    """Add the numbers in usage (nested dicts included) into total."""
    for key, value in (usage or {}).items():
        if isinstance(value, dict):
            add_usage(total.setdefault(key, {}), value)
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            total[key] = total.get(key, 0) + value
    return total


def request_hash(slug, prompt):
    """Hash of everything that shapes the answer: prompt, reasoning, providers."""
    settings = json.dumps({"reasoning": REASONING, "providers": MODELS[slug]}, sort_keys=True)
    return hashlib.sha256((prompt + "\0" + settings).encode("utf-8")).hexdigest()


def is_done(path, meta_path, req_hash):
    """True if the answer exists and its meta says it came from this request."""
    if not os.path.exists(path):
        return False
    try:
        with open(meta_path, encoding="utf-8") as f:
            return json.load(f).get("request_sha256") == req_hash
    except (OSError, ValueError):
        return False


def run_one(slug, prompt, api_key, out_dir, info, inputs):
    """Retry slug on one prompt until it answers with schema-valid JSON, then
    write the answer and its metadata (inputs included). Returns False if it
    gave up."""
    name = slug.replace("/", "__")
    path = os.path.join(out_dir, name + ".json")
    meta_path = os.path.join(out_dir, name + ".meta.json")
    failed_path = os.path.join(out_dir, name + ".failed.txt")
    tag = f"[{slug} | {os.path.basename(out_dir)}]"
    req_hash = request_hash(slug, prompt)
    if is_done(path, meta_path, req_hash):
        print(f"{tag} already done, skipping", flush=True)
        return True

    total_start, wait = time.time(), RETRY_WAIT
    usage_total, content = {}, None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        start = time.time()
        try:
            response = ask(slug, prompt, api_key)
            add_usage(usage_total, response.get("usage"))
            content = response["choices"][0]["message"]["content"]
            data = parse_json(content)
            check_schema(data)
            break
        except PermanentError:
            raise
        except Exception as e:
            print(f"{tag} attempt {attempt}/{MAX_ATTEMPTS} failed ({time.time() - start:.0f}s): {e}", flush=True)
            if attempt == MAX_ATTEMPTS:
                os.makedirs(out_dir, exist_ok=True)
                write_atomic(failed_path, f"Error after {attempt} attempts: {e}\n\n"
                                          f"Last raw answer received:\n{content if content is not None else '(none)'}\n")
                print(f"{tag} gave up -> {failed_path}", flush=True)
                return False
            if stop.wait(wait):
                return False
            wait = min(wait * 2, RETRY_MAX)

    # Answer first, meta last: is_done needs the meta's hash, so a kill in
    # between only means this file gets redone.
    choice = response["choices"][0]
    os.makedirs(out_dir, exist_ok=True)
    write_json(path, data)
    write_json(meta_path, {
        "model": slug,
        "served_model": response.get("model"),
        "generation_id": response.get("id"),
        "finish_reason": choice.get("finish_reason"),
        "provider": response.get("provider"),
        "providers_allowed": MODELS[slug],
        "reasoning_config": REASONING,
        "model_info": info,
        "request_sha256": req_hash,
        "usage": response.get("usage"),
        "usage_all_attempts": usage_total,
        "attempts": attempt,
        "seconds_last_attempt": round(time.time() - start, 1),
        "seconds_total": round(time.time() - total_start, 1),
        "reasoning_trace": choice["message"].get("reasoning"),
        **inputs,
        "prompt": prompt,
    })
    if os.path.exists(failed_path):
        os.remove(failed_path)
    print(f"{tag} ok after {attempt} attempt(s) ({time.time() - start:.0f}s) -> {path}", flush=True)
    return True


def run_model(slug, jobs, api_key, failed, catalog):
    """Work through every (prompt, out_dir, inputs) job for one model."""
    info = model_info(slug, catalog)
    for prompt, out_dir, inputs in jobs:
        if stop.is_set():
            return
        try:
            if not run_one(slug, prompt, api_key, out_dir, info, inputs):
                failed.append((slug, out_dir))
        except PermanentError as e:
            print(f"[{slug}] stopped, retrying will not help: {e}", flush=True)
            failed.append((slug, "all remaining files"))
            return


def main():
    args = parse_args()
    template = read(args.prompt)
    out_root = os.path.abspath(os.path.expanduser(args.output))
    jobs = []
    for g in grammar_files(args.grammar):
        grammar_text = read(g)
        jobs.append((build_prompt(template, grammar_text),
                     os.path.join(out_root, os.path.splitext(os.path.basename(g))[0]),
                     {"prompt_file": os.path.abspath(os.path.expanduser(args.prompt)),
                      "grammar_file": os.path.abspath(g),
                      "grammar_text": grammar_text}))

    catalog = fetch_catalog()
    failed = []
    threads = [threading.Thread(target=run_model, args=(slug, jobs, args.api_key, failed, catalog),
                                daemon=True)
               for slug in MODELS]
    for t in threads:
        t.start()
    try:
        # Join with a timeout so Ctrl-C reaches the main thread.
        while any(t.is_alive() for t in threads):
            for t in threads:
                t.join(timeout=0.5)
    except KeyboardInterrupt:
        stop.set()
        raise SystemExit("\nInterrupted; finished answers are saved, rerun to resume.")

    if failed:
        print("\nNot completed:")
        for slug, where in failed:
            print(f"  {slug}: {where}")


if __name__ == "__main__":
    main()
