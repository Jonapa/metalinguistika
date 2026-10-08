import argparse
import json
import sys
import time
from pathlib import Path

from openai import OpenAI, OpenAIError
from tqdm import tqdm

LATXA_URL = "http://trumoi.ixa.eus:8002/v1"
LATXA_MODEL = "Latxa-Llama-3.1-70B-Instruct-exp_2_101_v2-FP8"
LATXA_API_KEY = "EMPTY"
SLEEP_BETWEEN_QUERIES = 10  # seconds, to avoid overloading the server

CUSTOM_CHAT_TEMPLATE = r"""
{{- bos_token }}
{%- if custom_tools is defined %}
    {%- set tools = custom_tools %}
{%- endif %}
{%- if not tools_in_user_message is defined %}
    {%- set tools_in_user_message = true %}
{%- endif %}
{%- if not date_string is defined %}
    {%- if strftime_now is defined %}
        {%- set date_string = strftime_now("%Y-%m-%d") %}
    {% else %}
        {%- set date_string = "26 Jul 2024" %}
    {%- endif %}
{%- endif %}
{%- if not tools is defined %}
    {%- set tools = none %}
{%- endif %}

{%- if messages[0]['role'] == 'system' %}
    {%- set system_message = messages[0]['content']|trim %}
    {%- set messages = messages[1:] %}
{%- else %}
    {%- set system_message = "You are a helpful AI assistant called Latxa, created by HiTZ, the Basque Center for Language Technology (Hizkuntza Teknologiako Zentroa). \nYou are specifically designed to serve the Basque-speaking community, providing high-quality assistance in Euskara, though you can also communicate fluently in Spanish and English when needed.\n\nAlways respond in the same language the user writes to you. When addressed in Basque, use standard Euskara batua and maintain appropriate formality levels (hitano, zuka, berorika) based on the user's approach.\nTake special care with Basque grammar, vocabulary, and cultural nuances.\n" %}
{%- endif %}

{{- "<|start_header_id|>system<|end_header_id|>\n\n" }}
{%- if builtin_tools is defined or tools is not none %}
    {{- "Environment: ipython\n" }}
{%- endif %}
{%- if builtin_tools is defined %}
    {{- "Tools: " + builtin_tools | reject('equalto', 'code_interpreter') | join(", ") + "\n\n"}}
{%- endif %}
{%- if tools is not none and not tools_in_user_message %}
    {{- "You have access to the following functions. To call a function, please respond with JSON for a function call." }}
    {{- 'Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}.' }}
    {{- "Do not use variables.\n\n" }}
    {%- for t in tools %}
        {{- t | tojson(indent=4) }}
        {{- "\n\n" }}
    {%- endfor %}
{%- endif %}
{{- system_message }}
{{- "Today's date is " + date_string + "\n\n" }}
{{- "Cutting Knowledge Date: September 2025\n" }}
{{- "<|eot_id|>" }}

{%- if tools_in_user_message and not tools is none %}
    {%- if messages | length != 0 %}
        {%- set first_user_message = messages[0]['content']|trim %}
        {%- set messages = messages[1:] %}
    {%- else %}
        {{- raise_exception("Cannot put tools in the first user message when there's no first user message!") }}
{%- endif %}
    {{- '<|start_header_id|>user<|end_header_id|>\n\n' -}}
    {{- "Given the following functions, please respond with a JSON for a function call " }}
    {{- "with its proper arguments that best answers the given prompt.\n\n" }}
    {{- 'Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}.' }}
    {{- "Do not use variables.\n\n" }}
    {%- for t in tools %}
        {{- t | tojson(indent=4) }}
        {{- "\n\n" }}
    {%- endfor %}
    {{- first_user_message + "<|eot_id|>"}}
{%- endif %}

{%- for message in messages %}
    {%- if not (message.role == 'ipython' or message.role == 'tool' or 'tool_calls' in message) %}
        {{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' }}
    {%- elif 'tool_calls' in message %}
        {%- if not message.tool_calls|length == 1 %}
            {{- raise_exception("This model only supports single tool-calls at once!") }}
        {%- endif %}
        {%- set tool_call = message.tool_calls[0].function %}
        {%- if builtin_tools is defined and tool_call.name in builtin_tools %}
            {{- '<|start_header_id|>assistant<|end_header_id|>\n\n' -}}
            {{- "<|python_tag|>" + tool_call.name + ".call(" }}
            {%- for arg_name, arg_val in tool_call.arguments | items %}
                {{- arg_name + '="' + arg_val + '"' }}
                {%- if not loop.last %}
                    {{- ", " }}
                {%- endif %}
            {%- endfor %}
            {{- ")" }}
        {%- else  %}
            {{- '<|start_header_id|>assistant<|end_header_id|>\n\n' -}}
            {{- '{"name": "' + tool_call.name + '", ' }}
            {{- '"parameters": ' }}
            {{- tool_call.arguments | tojson }}
            {{- "}" }}
        {%- endif %}
        {%- if builtin_tools is defined %}
            {{- "<|eom_id|>" }}
        {%- else %}
            {{- "<|eot_id|>" }}
        {%- endif %}
    {%- elif message.role == "tool" or message.role == "ipython" %}
        {{- "<|start_header_id|>ipython<|end_header_id|>\n\n" }}
        {%- if message.content is mapping or message.content is iterable %}
            {{- message.content | tojson }}
        {%- else %}
            {{- message.content }}
        {%- endif %}
        {{- "<|eot_id|>" }}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|start_header_id|>assistant<|end_header_id|>\n\n' }}
{%- endif %}
"""

SYSTEM_PROMPT_DEFAULT = """
You are a helpful AI assistant called Latxa, created by HiTZ, the Basque Center for Language Technology (Hizkuntza Teknologiako Zentroa).
You are specifically designed to serve the Basque-speaking community, providing high-quality assistance in Euskara, though you can also communicate fluently in Spanish and English when needed.

Always respond in the same language the user writes to you. When addressed in Basque, use standard Euskara batua and maintain appropriate formality levels (hitano, zuka, berorika) based on the user's approach.
Take special care with Basque grammar, vocabulary, and cultural nuances.

Your capabilities include:
- Providing information on various general topics, as well as on Basque culture, history, language, and traditions.
- Understanding Basque better and producing cleaner Basque text compared to commercial systems.
- Translating long texts from several languages into Basque or vice versa.
- Explaining complex problems step by step.
- Writing and explaining code examples, if you ask me to.
- Processing user-provided text files (PDF, TXT, MD, CSV, DOC(X), XLS(X), PPT(X) or OpenOffice) (up to 10 MB).

Important limitations:
- Your knowledge is limited to information available up to September 2025.
- You cannot access the Internet or external tools.
- You cannot generate rich text documents (i.e., Word, Power Point, Excel), images, videos, and audio files.
- You cannot process images, videos, and audio files.
- You cannot run code or access external systems.

You should be honest and upfront about your capabilities and limitations if relevant for the user request.

For Basque-specific content (e.g., literature, bertsolaritza, toponymy, dialectal variations), provide as detailed and accurate information as possible while respecting linguistic diversity.

When asked about current events or time-sensitive information beyond your knowledge cutoff, acknowledge your limitations clearly. If users provide new information in their messages about current events in the Basque Country or elsewhere, you may reference and discuss it based on what they've shared.

Stylistic guidelines:
- When responding in Basque, maintain a warm, approachable tone with grammatically correct Euskara.
- Use paragraphs to structure your responses logically, but avoid using paragraph titles unless strictly necessary.
- Make sure to highlight just the relevant information using bold case ('**') to make your responses more readable.
- When explaining technical concepts, balance precision with accessibility.
- Include Basque-specific terms and expressions when appropriate to make conversations feel natural.
- Keep responses concise yet informative, avoiding unnecessary verbosity.
- For creative content requests, adapt your style to match the requested format (formal, casual, poetic, etc.).
- When writing code examples, include helpful comments in the language the user is using.
- If you are unsure about how to respond to a question, ask for clarification or additional context.
- At the end of each response, try to naturally suggest some potential directions for the conversation in a conversational way. These suggestions should flow naturally as part of your response, in a single sentence, and not labeled as follow-up questions.
""".strip()


def query_latxa(client, messages, max_tokens=32768, max_retries=3):
    for attempt in range(1, max_retries + 1):
        try:
            r = client.chat.completions.create(
                model=LATXA_MODEL,
                messages=messages,
                temperature=0.7,
                top_p=0.9,
                max_tokens=max_tokens,
                extra_body={"chat_template": CUSTOM_CHAT_TEMPLATE},
            )
            return r.choices[0].message.content
        except (OpenAIError, IndexError) as e:
            if attempt == max_retries:
                tqdm.write(f"Failed after {max_retries} attempts: {e}")
                return None
            time.sleep(2 ** attempt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    args = parser.parse_args()

    client = OpenAI(base_url=LATXA_URL, api_key=LATXA_API_KEY, timeout=600)

    # Resume from the output if it exists: rows without a response are (re)queried
    path = args.output_path if args.output_path.exists() else args.input_path
    rows = json.loads(path.read_text(encoding="utf-8"))
    args.output_path.parent.mkdir(parents=True, exist_ok=True)

    n_errors = 0
    for row in tqdm([row for row in rows if not row.get("infer", {}).get("infer_response")], desc=LATXA_MODEL):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_DEFAULT},
            {"role": "user", "content": row["question"]},
        ]
        response = query_latxa(client, messages)
        time.sleep(SLEEP_BETWEEN_QUERIES)
        if not response:
            n_errors += 1
            continue
        row["infer"] = {"infer_model": LATXA_MODEL, "infer_user_message": row["question"], "infer_response": response}
        # Save after every row so an interruption keeps the progress
        args.output_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"{args.output_path}: {n_errors} errors (rerun to retry them)")
    sys.exit(1 if n_errors else 0)
