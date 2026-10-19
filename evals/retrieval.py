"""
Step 3: retrieval metrics for every system, on dev or test.

    python -m evals.retrieval --subset dev
    python -m evals.retrieval --subset test --save

Metrics (a chunk is "relevant" if its grade is 1 or 2):
  Recall@k     share of the question's relevant chunks that are in the top k
  Precision@k  share of the top k that is relevant (how much noise the LLM has to read)
  MRR@k        1 / rank of the first relevant chunk
  nDCG@k       ranking quality that also uses the grade (2 counts more than 1)
"""
import argparse
import math

from evals.common import QRELS, RESULTS_DIR, load_json, questions_for, save_json
from evals.runs import RUNS_DIR
from evals.systems import SYSTEMS


def metrics_for_question(ranked: list[str], grades: dict[str, int], k: int) -> dict:
    top = ranked[:k]
    relevant = {cid for cid, g in grades.items() if g > 0}
    hits = [cid in relevant for cid in top]

    first = next((i for i, h in enumerate(hits, 1) if h), None)
    dcg = sum((2 ** grades.get(cid, 0) - 1) / math.log2(i + 1) for i, cid in enumerate(top, 1))
    ideal = sorted(grades.values(), reverse=True)[:k]
    idcg = sum((2 ** g - 1) / math.log2(i + 1) for i, g in enumerate(ideal, 1))

    return {
        "recall": sum(hits) / len(relevant) if relevant else 0.0,
        "precision": sum(hits) / k,
        "mrr": 1 / first if first else 0.0,
        "ndcg": dcg / idcg if idcg else 0.0,
        "unjudged": sum(cid not in grades for cid in top) / k,
    }


def evaluate_run(run: dict, qrels: dict, question_ids: list[str], k: int) -> dict:
    per_q = [metrics_for_question(run[qid], qrels.get(qid, {}), k) for qid in question_ids]
    return {m: sum(r[m] for r in per_q) / len(per_q) for m in per_q[0]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--subset", choices=["dev", "test", "all"], default="dev")
    p.add_argument("--k", type=int, nargs="+", default=[5, 8])
    p.add_argument("--save", action="store_true")
    a = p.parse_args()

    qrels = load_json(QRELS)
    qids = [q["id"] for q in questions_for(a.subset) if q["id"] in qrels]
    report = {"subset": a.subset, "n_questions": len(qids), "systems": {}}

    print(f"\n{a.subset}: {len(qids)} questions")
    for k in a.k:
        print(f"\n{'system':24} {'R@'+str(k):>7} {'P@'+str(k):>7} {'MRR':>7} {'nDCG':>7} {'unjudged':>9} {'ms/q':>6}")
        for s in SYSTEMS:
            run_file = RUNS_DIR / f"{s.name}.json"
            if not run_file.exists():
                continue
            data = load_json(run_file)
            m = evaluate_run(data["run"], qrels, qids, k)
            report["systems"].setdefault(s.name, {"description": s.description, "ms_per_query": data["ms_per_query"]})[f"@{k}"] = m
            print(f"{s.name:24} {m['recall']:7.3f} {m['precision']:7.3f} {m['mrr']:7.3f} {m['ndcg']:7.3f} "
                  f"{m['unjudged']:9.1%} {data['ms_per_query']:6.0f}")

    if a.save:
        save_json(RESULTS_DIR / f"retrieval_{a.subset}.json", report)


if __name__ == "__main__":
    main()
