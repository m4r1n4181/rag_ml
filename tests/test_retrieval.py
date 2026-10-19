from src.ingestion.pdf_loader import Document
from src.retrieval.lexical import BM25Index, stem, strip_diacritics, tokenize
from src.retrieval.vector_store import chunk_id, reciprocal_rank_fusion


def test_strip_diacritics():
    assert strip_diacritics("čćžšđ") == "cczsdj"


def test_inflected_forms_share_a_token():
    assert len({stem(w) for w in ["model", "modela", "modelom", "modelima"]}) == 1
    assert len(set(tokenize("regularizacija regularizacije regularizacioni"))) == 1


def test_short_words_are_kept():
    assert tokenize("L1 i L2 norma") == ["l1", "l2", "norm"]


def test_bm25_finds_keyword():
    index = BM25Index(
        ["a", "b", "c"],
        ["Lasso daje retke modele", "Ridge regresija koristi L2 normu", "kNN klasifikator"],
        [{}, {}, {}],
    )
    assert index.search("lasso", 1)[0]["id"] == "a"


def test_rrf_rewards_agreement_between_lists():
    vector = [{"id": "x"}, {"id": "y"}, {"id": "z"}]
    keyword = [{"id": "y"}, {"id": "w"}]
    fused = reciprocal_rank_fusion([vector, keyword])
    assert fused[0]["id"] == "y"
    assert {h["id"] for h in fused} == {"x", "y", "z", "w"}


def test_chunk_ids_are_stable_and_readable():
    slide = Document("t", {"source": "L02.pdf", "slide": 28, "type": "presentation"})
    page = Document("t", {"source": "MU.pdf", "page": 45, "chunk": 2, "type": "textbook"})
    assert chunk_id(slide) == "L02.pdf::s28"
    assert chunk_id(page) == "MU.pdf::p45_c2"
