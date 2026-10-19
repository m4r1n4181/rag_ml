from evals.common import make_split
from evals.human_check import cohens_kappa
from evals.retrieval import metrics_for_question


def test_metrics_perfect_ranking():
    m = metrics_for_question(["a", "b", "c"], {"a": 2, "b": 1, "c": 0}, k=2)
    assert m["recall"] == 1.0 and m["precision"] == 1.0 and m["mrr"] == 1.0 and m["ndcg"] == 1.0


def test_metrics_first_relevant_at_rank_two():
    m = metrics_for_question(["x", "a"], {"a": 2, "x": 0}, k=2)
    assert m["mrr"] == 0.5 and m["precision"] == 0.5 and m["recall"] == 1.0 and m["ndcg"] < 1.0


def test_unjudged_chunks_are_counted():
    assert metrics_for_question(["new", "a"], {"a": 1}, k=2)["unjudged"] == 0.5


def test_split_is_stratified_and_disjoint():
    questions = [{"id": f"{s}.{i}"} for s in ["I", "II", "III"] for i in range(1, 7)]
    split = make_split(questions, dev_share=1 / 3, seed=1)
    assert not set(split["dev"]) & set(split["test"])
    assert len(split["dev"]) + len(split["test"]) == 18
    for section in ["I", "II", "III"]:
        assert sum(q.startswith(section + ".") for q in split["dev"]) == 2


def test_kappa():
    assert cohens_kappa(["yes", "no", "yes"], ["yes", "no", "yes"]) == 1.0
    assert cohens_kappa(["yes", "no"], ["no", "yes"]) < 0
