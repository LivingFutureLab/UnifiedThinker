#!/usr/bin/env python3
"""
RISEBench evaluation with GPT-4o as judge (OpenAI Chat Completions API).

This is a thin wrapper around the shipped gpt_eval.py: it reuses that module's
eval_vanilla / extract / calculate_score / aggregation / main() verbatim and only
overrides the API-call layer. gpt_eval.py itself is left untouched.

The upstream gpt_eval.py had api_key='' / api_base='' hard-coded. RISEBench's paper
uses a GPT-family judge, so this wrapper points the same OpenAI-protocol call at
GPT-4o. Any OpenAI-compatible endpoint works (official API, or a compatible gateway
by overriding OPENAI_API_BASE); the request/response schema is the standard
`messages` in / `choices[]` out.

Config comes from env vars (no secrets in source):
    OPENAI_API_KEY    (required)
    OPENAI_API_BASE   (default: https://api.openai.com/v1/chat/completions;
                       override to use an OpenAI-compatible gateway)
    OPENAI_MODEL      (default: gpt-4o-2024-08-06 — the snapshot benchmarks pin to)

Usage (from the RISEBench script dir):
    OPENAI_API_KEY=sk-... \
    python3 gpt4o_eval.py \
        --data   <path>/datav2_total_w_subtask.json \
        --input  <path>/data \
        --output <path>/outputs/<MODEL_NAME> \
        --prefix eval_risebench_by_ \
        --nproc  8
"""
import os
import json
import time

import requests

import gpt_eval
from gpt_eval import main  # reuse the upstream driver unchanged


API_KEY = os.environ.get("OPENAI_API_KEY", "")
API_BASE = os.environ.get(
    "OPENAI_API_BASE",
    "https://api.openai.com/v1/chat/completions",
)
MODEL_NAME = os.environ.get("OPENAI_MODEL", "gpt-4o-2024-08-06")


def gpt_generate(inputs, model=MODEL_NAME, temperature=0, max_tokens=4096,
                 image_size=768, **kwargs):
    """Drop-in replacement for gpt_eval.gpt_generate.

    Same OpenAI request/response schema as upstream; only the endpoint,
    auth key and default model differ. prepare_inputs() (from utils) already
    produces OpenAI-style `messages` with base64 image_url parts, so nothing
    about the message construction changes.
    """
    input_msgs = gpt_eval.prepare_inputs(inputs, image_size=image_size)
    temperature = kwargs.pop("temperature", temperature)
    max_tokens = kwargs.pop("max_tokens", max_tokens)
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}",
    }
    payload = dict(
        model=model,
        messages=input_msgs,
        max_tokens=max_tokens,
        n=1,
        temperature=temperature,
        **kwargs,
    )

    retries = 10
    response = None
    for attempt in range(1, retries + 1):
        try:
            response = requests.post(
                API_BASE, headers=headers, data=json.dumps(payload), timeout=120)
            break
        except Exception as e:
            print(f"❌ [Attempt {attempt}/{retries}] request error: {e}")
            if attempt == retries:
                raise
            time.sleep(3)

    ret_code = response.status_code
    ret_code = 0 if (200 <= int(ret_code) < 300) else ret_code
    answer = "Failed to obtain answer via API. "
    try:
        resp_struct = json.loads(response.text)
        answer = resp_struct["choices"][0]["message"]["content"].strip()
    except Exception as err:
        print(f"{type(err)}: {err}")
        print(response.text if hasattr(response, "text") else response)

    return ret_code, answer, response


def _install_overrides():
    if not API_KEY:
        raise SystemExit(
            "OPENAI_API_KEY is not set. Export it before running, e.g.\n"
            "  OPENAI_API_KEY=sk-... python3 gpt4o_eval.py ...")
    # eval_vanilla and main() reference these as module globals on gpt_eval,
    # so rebind them there (not just here) before the driver runs.
    gpt_eval.gpt_generate = gpt_generate
    gpt_eval.api_key = API_KEY
    gpt_eval.api_base = API_BASE
    gpt_eval.MODEL_NAME = MODEL_NAME
    print(f"[gpt4o_eval] judge model = {MODEL_NAME}")
    print(f"[gpt4o_eval] endpoint    = {API_BASE}")


if __name__ == "__main__":
    _install_overrides()
    main()
