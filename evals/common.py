"""Shared helpers for the evaluation scripts: data files, the dev/test split and LLM calls."""
import json
import os
import random
import re
import time
from collections import defaultdict
from pathlib import Path

import anthropic

from src.config import settings

EVAL_DIR = Path("data/eval")
QUESTIONS = EVAL_DIR / "questions.json"
SPLIT = EVAL_DIR / "split.json"
QRELS = EVAL_DIR / "qrels.json"
RESULTS_DIR = EVAL_DIR / "results"

JUDGE_MODEL = os.getenv("JUDGE_MODEL", "claude-sonnet-5-5")

# USD per million tokens (input, output). Used only to stop runaway spending.
PRICES = {
    "claude-haiku-4-5-20251001": (1.0, 5.0),
    "claude-sonnet-5-5": (3.0, 15.0),
}
MAX_SPEND_USD = float(os.getenv("MAX_EVAL_SPEND_USD", "15"))


def load_questions() -> list[dict]:
    return json.loads(QUESTIONS.read_text(encoding="utf-8"))


def load_json(path: Path, default=None):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def save_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")


def make_split(questions: list[dict], dev_share: float = 1 / 3, seed: int = 42) -> dict:
    """Stratified by exam section so dev and test both cover every topic."""
    by_section = defaultdict(list)
    for q in questions:
        by_section[q["id"].split(".")[0]].append(q["id"])

    rng = random.Random(seed)
    dev, test = [], []
    for ids in by_section.values():
        ids = sorted(ids)
        rng.shuffle(ids)
        n_dev = max(1, round(len(ids) * dev_share))
        dev += ids[:n_dev]
        test += ids[n_dev:]
    return {"seed": seed, "dev": sorted(dev), "test": sorted(test)}


def get_split() -> dict:
    split = load_json(SPLIT)
    if split is None:
        split = make_split(load_questions())
        save_json(SPLIT, split)
    return split


def questions_for(subset: str) -> list[dict]:
    questions = load_questions()
    if subset == "all":
        return questions
    ids = set(get_split()[subset])
    return [q for q in questions if q["id"] in ids]


class SpendTracker:
    def __init__(self):
        self.usd = 0.0

    def add(self, model: str, usage) -> None:
        price_in, price_out = PRICES.get(model, (3.0, 15.0))
        self.usd += (usage.input_tokens * price_in + usage.output_tokens * price_out) / 1e6
        if self.usd > MAX_SPEND_USD:
            raise RuntimeError(f"Spending limit reached (${self.usd:.2f}). Raise MAX_EVAL_SPEND_USD to continue.")


spend = SpendTracker()
_client = None


def client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=settings.anthropic_api_key, max_retries=5)
    return _client


def call_llm(prompt: str, model: str, system: str | None = None, max_tokens: int = 4000) -> str:
    # No temperature: newer models (the judge) reject sampling parameters.
    kwargs = dict(model=model, max_tokens=max_tokens, messages=[{"role": "user", "content": prompt}])
    if system:
        kwargs["system"] = system
    for attempt in range(3):
        try:
            msg = client().messages.create(**kwargs)
            spend.add(model, msg.usage)
            # The judge model may return thinking blocks before the answer
            return "".join(block.text for block in msg.content if block.type == "text")
        except anthropic.APIStatusError as e:
            if attempt == 2 or e.status_code < 500:
                raise
            time.sleep(5 * (attempt + 1))


def call_llm_json(prompt: str, model: str = JUDGE_MODEL, system: str | None = None, max_tokens: int = 4000):
    """Ask for JSON and parse the first JSON object in the reply."""
    text = call_llm(prompt, model, system, max_tokens)
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON in model reply: {text[:200]}")
    return json.loads(match.group(0))
