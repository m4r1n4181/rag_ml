"""
Step 2: build relevance labels (qrels) by pooling.

For each question we take the union of the top-10 chunks from every system (the "pool"),
shuffle it so the judge can't tell which system found what, and grade each chunk
against the question and its reference answer:

    2 = contains a substantial part of the expected answer
    1 = related and partly useful (a definition, one piece of the answer)
    0 = not useful for this question

Chunks outside the pool count as not relevant. That is the standard pooling assumption
(TREC). Every metric report also prints how many retrieved chunks were unjudged.

    python -m evals.label
"""
import json
import random

from loguru import logger

from evals.common import JUDGE_MODEL, QRELS, call_llm_json, load_json, load_questions, save_json, spend
from evals.runs import RUNS_DIR, TOP_K

PROMPT = """You are grading search results for a university course assistant (course: Machine Learning 1, in Serbian).

Question:
<question>
{question}
</question>

Reference answer from the course study guide:
<reference_answer>
{reference}
</reference_answer>

Below are {n} retrieved passages (lecture slides or textbook pages). Text was extracted from PDFs,
so formulas may be broken. Grade each passage for how useful it is for answering THIS question:

2 = contains a substantial part of the expected answer (key definitions, formulas or explanations the answer needs)
1 = related and partly useful (touches the topic, contains one small piece of the answer, or a needed definition)
0 = not useful for this question (different topic, or only a table of contents / title)

{passages}

Reply with JSON only, mapping every passage label to its grade, e.g. {{"P1": 0, "P2": 2}}."""


def load_pool() -> tuple[dict[str, set[str]], dict[str, dict]]:
    pool, texts = {}, {}
    for run_file in sorted(RUNS_DIR.glob("*.json")):
        run = load_json(run_file)["run"]
        for qid, ids in run.items():
            pool.setdefault(qid, set()).update(ids[:TOP_K])
    for chunk_file in sorted((RUNS_DIR / "chunks").glob("*.json")):
        for cid, chunk in load_json(chunk_file).items():
            # Prefer the text from the shipped index (it has the final cleaning)
            if cid not in texts or chunk_file.stem == "e5+header":
                texts[cid] = chunk
    return pool, texts


def describe(chunk: dict) -> str:
    meta = chunk["metadata"]
    where = f"slide {meta.get('slide')}" if meta.get("type") == "presentation" else f"textbook page {meta.get('page')}"
    return f"{meta.get('source')}, {where}"


def label_question(q: dict, chunk_ids: list[str], texts: dict) -> dict[str, int]:
    ids = sorted(chunk_ids)
    random.Random(q["id"]).shuffle(ids)
    passages = "\n\n".join(
        f"<passage label=\"P{i}\" source=\"{describe(texts[cid])}\">\n{texts[cid]['content'][:1500]}\n</passage>"
        for i, cid in enumerate(ids, 1)
    )
    prompt = PROMPT.format(question=q["question"], reference=q["reference_answer"][:4000], n=len(ids), passages=passages)
    grades = call_llm_json(prompt, JUDGE_MODEL)
    return {cid: int(grades.get(f"P{i}", 0)) for i, cid in enumerate(ids, 1)}


def main():
    pool, texts = load_pool()
    qrels = load_json(QRELS, default={})
    for q in load_questions():
        todo = [cid for cid in pool.get(q["id"], ()) if cid not in qrels.get(q["id"], {})]
        if not todo:
            continue
        qrels.setdefault(q["id"], {}).update(label_question(q, todo, texts))
        save_json(QRELS, qrels)
        n_rel = sum(g > 0 for g in qrels[q["id"]].values())
        logger.info(f"{q['id']}: {len(todo)} graded, {n_rel} relevant, spent ${spend.usd:.2f}")
    logger.success(f"Done. {sum(len(v) for v in qrels.values())} judgments, ${spend.usd:.2f}")


if __name__ == "__main__":
    main()
