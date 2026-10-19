"""
Step 5: check the LLM judge against a human (you).

Shows a random sample of judged answers. For every key point you type y / p / n
(covered / partial / missing). At the end it prints how often you and the judge agree
and Cohen's kappa (agreement corrected for chance). Your grades are saved, so you can
stop and continue later.

    python -m evals.human_check --n 15
"""
import argparse
import random
from collections import Counter

from evals.answers import ANSWERS_DIR, KEY_POINTS
from evals.common import RESULTS_DIR, load_json, save_json

HUMAN = ANSWERS_DIR / "human_grades.json"
KEYS = {"y": "yes", "p": "partial", "n": "no"}


def cohens_kappa(a: list[str], b: list[str]) -> float:
    n = len(a)
    observed = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    expected = sum(ca[c] * cb[c] for c in set(a) | set(b)) / (n * n)
    return (observed - expected) / (1 - expected) if expected < 1 else 1.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=15)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    key_points = load_json(KEY_POINTS)
    human = load_json(HUMAN, default={})
    items = []
    for pipeline in ("original", "current"):
        answers = load_json(ANSWERS_DIR / f"{pipeline}.json", default={})
        judged = load_json(ANSWERS_DIR / f"{pipeline}_judged.json", default={})
        items += [(pipeline, qid) for qid in judged if qid in answers]
    random.Random(a.seed).shuffle(items)

    for pipeline, qid in items[: a.n]:
        key = f"{pipeline}:{qid}"
        if key in human:
            continue
        answer = load_json(ANSWERS_DIR / f"{pipeline}.json")[qid]["answer"]
        print("\n" + "=" * 80 + f"\nQuestion {qid}\n" + "-" * 80 + f"\n{answer}\n" + "-" * 80)
        grades = []
        for i, point in enumerate(key_points[qid], 1):
            reply = ""
            while reply not in KEYS:
                reply = input(f"K{i}: {point}\n   covered? [y/p/n] ").strip().lower()
            grades.append(KEYS[reply])
        human[key] = grades
        save_json(HUMAN, human)

    ours, theirs = [], []
    for key, grades in human.items():
        pipeline, qid = key.split(":")
        judge_grades = load_json(ANSWERS_DIR / f"{pipeline}_judged.json")[qid]["coverage"]
        ours += grades
        theirs += judge_grades[: len(grades)]

    if ours:
        agreement = sum(x == y for x, y in zip(ours, theirs)) / len(ours)
        kappa = cohens_kappa(ours, theirs)
        print(f"\n{len(human)} answers, {len(ours)} key points")
        print(f"Agreement with judge: {agreement:.1%}   Cohen's kappa: {kappa:.2f}")
        save_json(RESULTS_DIR / "judge_agreement.json",
                  {"answers": len(human), "key_points": len(ours), "agreement": agreement, "kappa": kappa})


if __name__ == "__main__":
    main()
