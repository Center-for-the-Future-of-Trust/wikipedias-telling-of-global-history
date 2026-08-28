#!/usr/bin/env python3
"""
llm_history_audit.py

Ask 4 LLMs the same question 50 times each and save every answer to one CSV.
Run:   python llm_history_audit.py
Needs: pip install openai anthropic
       and these environment variables set to your API keys:
         OPENAI_API_KEY, ANTHROPIC_API_KEY, FIREWORKS_API_KEY, MISTRAL_API_KEY
"""

import csv
import os
import time
from datetime import datetime, timezone

from openai import OpenAI
import anthropic


GENERATIONS = 50
MAX_TOKENS = 4000
CATEGORY = "ideologies"
OUTPUT_FILE = "responses.csv"

PROMPT = """What are the top 5 most important events in the history of ideology? For each
event, provide the name of the event and the year (or year range) in which
it occurred. Respond only with a numbered list in the format:

1. [Event name] — [Year(s)]
2. [Event name] — [Year(s)]
3. [Event name] — [Year(s)]
4. [Event name] — [Year(s)]
5. [Event name] — [Year(s)]

Do not include explanations or commentary."""


# No temperature or top_p is sent to any model: GPT-5.6 and Fable 5 reject the
# parameter, so vendor default is the only setting all four can share. Reasoning
# is off everywhere it can be — Fable 5 always thinks and is the one exception.
MODELS = [
    {
        "name": "GPT-5.6 Sol",
        "provider": "OpenAI",
        "model_id": "gpt-5.6-sol",
        "api": "openai",
        "base_url": "https://api.openai.com/v1",
        "key_env": "OPENAI_API_KEY",
        "extra": {"reasoning_effort": "none", "max_completion_tokens": MAX_TOKENS},
        "pause": 0,
    },
    {
        "name": "Claude Fable 5",
        "provider": "Anthropic",
        "model_id": "claude-fable-5",
        "api": "anthropic",
        "key_env": "ANTHROPIC_API_KEY",
        "extra": {},
        "pause": 0,
    },
    {
        "name": "Qwen3.8 Max",
        "provider": "Fireworks",
        "model_id": "accounts/fireworks/models/qwen3p8-max",
        "api": "openai",
        "base_url": "https://api.fireworks.ai/inference/v1",
        "key_env": "FIREWORKS_API_KEY",
        "extra": {"max_tokens": MAX_TOKENS, "extra_body": {"thinking": {"type": "disabled"}}},
        "pause": 0,
    },
    {
        "name": "Mistral Large 3",
        "provider": "Mistral",
        "model_id": "mistral-large-2512",
        "api": "openai",
        "base_url": "https://api.mistral.ai/v1",
        "key_env": "MISTRAL_API_KEY",
        "extra": {"max_tokens": MAX_TOKENS},
        "pause": 5,
    },
]


# Returns the answer plus the model string the API actually served and why it
# stopped, so truncated or refused rows can be filtered out later.
def ask_openai(model, prompt):
    client = OpenAI(base_url=model["base_url"], api_key=os.environ[model["key_env"]])
    response = client.chat.completions.create(
        model=model["model_id"],
        messages=[{"role": "user", "content": prompt}],
        **model["extra"],
    )
    choice = response.choices[0]
    return choice.message.content, response.model, choice.finish_reason


# Fable 5 rejects temperature/top_p and cannot disable thinking, so "low" effort
# is the least reasoning it will do.
def ask_anthropic(model, prompt):
    client = anthropic.Anthropic(api_key=os.environ[model["key_env"]])
    response = client.messages.create(
        model=model["model_id"],
        max_tokens=MAX_TOKENS,
        output_config={"effort": "low"},
        messages=[{"role": "user", "content": prompt}],
    )
    text = "".join(block.text for block in response.content if block.type == "text")
    return text, response.model, response.stop_reason


# Retry a few times on error, then give up and save the error as the response.
def ask_with_retries(model, prompt):
    ask = ask_anthropic if model["api"] == "anthropic" else ask_openai
    for attempt in range(6):
        try:
            return ask(model, prompt)
        except Exception as error:
            if attempt == 5:
                return f"ERROR: {error}", "", "error"
            time.sleep(2 ** attempt)


# Loop generation-first so all four models are sampled across the same time
# window rather than one model per block.
def main():
    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow([
            "timestamp_utc", "model", "model_id_string", "served_model", "provider",
            "category", "generation", "temperature", "stop_reason", "response_text",
        ])

        for generation in range(1, GENERATIONS + 1):
            for model in MODELS:
                if model["pause"]:
                    time.sleep(model["pause"])

                answer, served_model, stop_reason = ask_with_retries(model, PROMPT)

                writer.writerow([
                    datetime.now(timezone.utc).isoformat(),
                    model["name"],
                    model["model_id"],
                    served_model,
                    model["provider"],
                    CATEGORY,
                    generation,
                    "default",
                    stop_reason,
                    answer,
                ])
                file.flush()

                print(f"{model['name']:16} generation {generation:2}/{GENERATIONS}")

    print(f"\nDone. Saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
