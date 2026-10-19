from src.generation import rag_chain
from src.generation.prompts import build_prompt, format_source


def test_prompt_contains_sources_without_similarity_scores():
    chunks = [{"content": "Lasso ...", "similarity": 0.41,
               "metadata": {"type": "presentation", "lecture": "Linearna regresija", "slide": 28}}]
    prompt = build_prompt("Šta je Lasso?", chunks)
    assert "Predavanje: Linearna regresija, slajd 28" in prompt
    assert "0.41" not in prompt


def test_format_source_textbook():
    assert format_source({"type": "textbook", "page": 45}) == "Praktikum, strana 45"


def test_follow_up_detection():
    assert rag_chain.looks_like_follow_up("a kod Lasso?")
    assert rag_chain.looks_like_follow_up("Zašto je to tako?")
    assert not rag_chain.looks_like_follow_up(
        "Objasniti pojam regularizacije u kontekstu linearne regresije i navesti načine primene")


def test_rewrite_skipped_without_history(monkeypatch):
    monkeypatch.setattr(rag_chain, "generate", lambda *a, **k: (_ for _ in ()).throw(AssertionError("called")))
    assert rag_chain.rewrite_query("a kod Lasso?", history=[]) == "a kod Lasso?"


def test_rewrite_rejects_multiline_output(monkeypatch):
    monkeypatch.setattr(rag_chain, "generate", lambda *a, **k: "Evo pitanja:\nKako radi Lasso?")
    history = [{"role": "user", "content": "Šta je Ridge?"}, {"role": "assistant", "content": "..."}]
    assert rag_chain.rewrite_query("a Lasso?", history) == "a Lasso?"


def test_rewrite_uses_model_output(monkeypatch):
    monkeypatch.setattr(rag_chain, "generate", lambda *a, **k: "Kako radi Lasso regularizacija?")
    history = [{"role": "user", "content": "Šta je Ridge?"}, {"role": "assistant", "content": "..."}]
    assert rag_chain.rewrite_query("a Lasso?", history) == "Kako radi Lasso regularizacija?"
