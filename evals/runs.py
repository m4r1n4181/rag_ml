"""
Step 1: run every system on every question and save the top-k chunk ids.

    python -m evals.runs            # all systems
    python -m evals.runs e5+header  # one system
"""
import sys
import time

from loguru import logger

from evals.common import EVAL_DIR, load_questions, save_json
from evals.systems import BY_NAME, SYSTEMS, activate, retrieve

RUNS_DIR = EVAL_DIR / "runs"
TOP_K = 10


def run_system(system) -> None:
    collection = activate(system)
    questions = load_questions()
    retrieve(system, collection, questions[0]["question"], TOP_K)  # warm-up: model loading is not query latency
    run, texts, start = {}, {}, time.time()
    for q in questions:
        hits = retrieve(system, collection, q["question"], TOP_K)
        run[q["id"]] = [h["id"] for h in hits]
        texts.update({h["id"]: {"content": h["content"], "metadata": h["metadata"]} for h in hits})
    ms_per_query = (time.time() - start) / len(questions) * 1000
    save_json(RUNS_DIR / f"{system.name}.json", {"system": system.name, "ms_per_query": ms_per_query, "run": run})
    save_json(RUNS_DIR / "chunks" / f"{system.name}.json", texts)
    logger.success(f"{system.name}: {len(run)} questions, {ms_per_query:.0f} ms/query")


if __name__ == "__main__":
    names = sys.argv[1:] or [s.name for s in SYSTEMS]
    for name in names:
        run_system(BY_NAME[name])
