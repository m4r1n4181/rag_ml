"""
Step 4: end-to-end answer quality, original pipeline vs current pipeline.

For each question:
  1. The reference answer is split once into key points (shared by all pipelines).
  2. Each pipeline retrieves context and generates an answer with the app's LLM.
  3. A stronger judge model checks which key points the answer covers (completeness)
     and which statements in the answer are not supported by the retrieved context
     (faithfulness, i.e. hallucination check).

    python -m evals.answers --subset test
    python -m evals.answers --subset dev --pipelines current_k5 current_k8 current_k10

Results are cached in data/eval/answers/, so re-running only fills in what is missing.
"""
import argparse

from loguru import logger

from evals import baseline_v1
from evals.common import EVAL_DIR, JUDGE_MODEL, RESULTS_DIR, call_llm_json, load_json, questions_for, save_json, spend
from evals.systems import BY_NAME, activate, retrieve
from src.config import settings
from src.generation.llm import generate
from src.generation.prompts import SYSTEM_PROMPT, build_prompt

ANSWERS_DIR = EVAL_DIR / "answers"
KEY_POINTS = EVAL_DIR / "reference_key_points.json"

PIPELINES = {
    "original": {"system": "v1", "k": baseline_v1.N_RESULTS,
                 "system_prompt": baseline_v1.SYSTEM_PROMPT, "build_prompt": baseline_v1.build_prompt},
    "current": {"system": "e5+header", "k": settings.n_results,
                "system_prompt": SYSTEM_PROMPT, "build_prompt": build_prompt},
}

KEY_POINTS_PROMPT = """Below is an exam question from a Machine Learning course (in Serbian) and a reference answer.
List the key points a complete exam answer must contain: 3 to 8 short, self-contained statements,
each one checkable on its own (a definition, a formula, a property, a reason). Write them in Serbian.

<question>{question}</question>
<reference_answer>{reference}</reference_answer>

Reply with JSON only: {{"key_points": ["...", "..."]}}"""

JUDGE_PROMPT = """You are grading a study assistant's answer to an exam question (Machine Learning course, Serbian).

<question>{question}</question>

<key_points>
{key_points}
</key_points>

<prompt_given_to_the_assistant>
{context}
</prompt_given_to_the_assistant>

<answer>
{answer}
</answer>

Task 1 (completeness): for each key point, decide if the answer covers it:
"yes" = stated correctly, "partial" = mentioned but incomplete or vague, "no" = missing or wrong.
Judge the meaning, not the wording.

Task 2 (faithfulness): split the answer into its factual statements and list the ones that are NOT
supported by the context in the prompt above (even if they are true in general). Citations of a
source label that appears in the prompt (e.g. "slajd 26") count as supported. Ignore structure and
statements that only say something is missing from the materials.

Reply with JSON only:
{{"coverage": {{"K1": "yes|partial|no", ...}}, "n_statements": <int>, "unsupported": ["...", "..."]}}"""

COVERAGE_SCORE = {"yes": 1.0, "partial": 0.5, "no": 0.0}


def pipeline_config(name: str) -> dict:
    """'original', 'current', or 'current_k5' / 'current_k10' for choosing k on dev."""
    if name.startswith("current_k"):
        return {**PIPELINES["current"], "k": int(name.removeprefix("current_k"))}
    return PIPELINES[name]


def get_key_points(questions: list[dict]) -> dict:
    cache = load_json(KEY_POINTS, default={})
    for q in questions:
        if q["id"] not in cache:
            reply = call_llm_json(KEY_POINTS_PROMPT.format(question=q["question"], reference=q["reference_answer"]))
            cache[q["id"]] = reply["key_points"]
            save_json(KEY_POINTS, cache)
    return cache


def generate_answers(name: str, questions: list[dict]) -> dict:
    path = ANSWERS_DIR / f"{name}.json"
    cache = load_json(path, default={})
    todo = [q for q in questions if "prompt" not in cache.get(q["id"], {})]
    if not todo:
        return cache

    cfg = pipeline_config(name)
    system = BY_NAME[cfg["system"]]
    collection = activate(system)
    for q in todo:
        chunks = retrieve(system, collection, q["question"], cfg["k"])
        prompt = cfg["build_prompt"](q["question"], chunks)
        entry = cache.get(q["id"], {})
        if "answer" not in entry:
            entry["answer"] = generate(prompt, cfg["system_prompt"])
            logger.info(f"[{name}] answered {q['id']}  (${spend.usd:.2f})")
        # The judge must see exactly what the model saw, including the source labels.
        entry.update({"context_ids": [c["id"] for c in chunks], "prompt": prompt})
        entry.pop("context", None)
        cache[q["id"]] = entry
        save_json(path, cache)
    return cache


def judge(name: str, questions: list[dict], answers: dict, key_points: dict) -> dict:
    path = ANSWERS_DIR / f"{name}_judged.json"
    cache = load_json(path, default={})
    for q in questions:
        if q["id"] in cache:
            continue
        points = key_points[q["id"]]
        a = answers[q["id"]]
        reply = call_llm_json(JUDGE_PROMPT.format(
            question=q["question"],
            key_points="\n".join(f"K{i}: {p}" for i, p in enumerate(points, 1)),
            context=a["prompt"][:40000],
            answer=a["answer"],
        ))
        coverage = [reply["coverage"].get(f"K{i}", "no") for i in range(1, len(points) + 1)]
        n_statements = max(int(reply.get("n_statements", 0)), 1)
        cache[q["id"]] = {
            "coverage": coverage,
            "completeness": sum(COVERAGE_SCORE.get(c, 0.0) for c in coverage) / len(coverage),
            "unsupported": reply.get("unsupported", []),
            "faithfulness": 1 - min(len(reply.get("unsupported", [])), n_statements) / n_statements,
        }
        save_json(path, cache)
        logger.info(f"[{name}] judged {q['id']}: completeness {cache[q['id']]['completeness']:.2f}")
    return cache


def summarize(judged: dict, qids: list[str]) -> dict:
    rows = [judged[q] for q in qids]
    n = len(rows)
    return {
        "n": n,
        "completeness": sum(r["completeness"] for r in rows) / n,
        "fully_complete": sum(r["completeness"] == 1.0 for r in rows) / n,
        "faithfulness": sum(r["faithfulness"] for r in rows) / n,
        "answers_with_unsupported_claims": sum(bool(r["unsupported"]) for r in rows) / n,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--subset", choices=["dev", "test", "all"], default="test")
    p.add_argument("--pipelines", nargs="+", default=list(PIPELINES),
                   help="original, current, or current_k<N> to try another k")
    a = p.parse_args()

    questions = questions_for(a.subset)
    key_points = get_key_points(questions)
    report = {"subset": a.subset, "judge_model": JUDGE_MODEL, "answer_model": settings.llm_model, "pipelines": {}}

    print(f"\n{'pipeline':10} {'complete':>9} {'fully':>7} {'faithful':>9} {'w/ unsupp.':>11}")
    for name in a.pipelines:
        answers = generate_answers(name, questions)
        judged = judge(name, questions, answers, key_points)
        s = summarize(judged, [q["id"] for q in questions])
        cfg = pipeline_config(name)
        report["pipelines"][name] = {"system": cfg["system"], "k": cfg["k"], **s}
        print(f"{name:10} {s['completeness']:9.3f} {s['fully_complete']:7.1%} {s['faithfulness']:9.3f} "
              f"{s['answers_with_unsupported_claims']:11.1%}")

    save_json(RESULTS_DIR / f"answers_{a.subset}.json", report)
    print(f"\nLLM spend this run: ${spend.usd:.2f}")


if __name__ == "__main__":
    main()
