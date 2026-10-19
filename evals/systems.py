"""
Retrieval configurations compared in the ablation study.

Each ablation index is built once with build_db.py and the matching env vars, e.g.
    CHROMA_PATH=data/ablations/minilm_noheader EMBEDDING_MODEL=paraphrase-multilingual-MiniLM-L12-v2 \
    USE_CONTEXT_HEADER=false python build_db.py
"""
from dataclasses import dataclass

from src.config import settings
from src.retrieval import vector_store

MINILM = "paraphrase-multilingual-MiniLM-L12-v2"
E5 = "intfloat/multilingual-e5-large-instruct"


@dataclass(frozen=True)
class System:
    name: str
    embedding_model: str
    chroma_path: str
    hybrid: bool
    description: str


SYSTEMS = [
    System("minilm", MINILM, "data/ablations/minilm_noheader", False, "MiniLM, vector only (original setup)"),
    System("minilm+bm25", MINILM, "data/ablations/minilm_noheader", True, "MiniLM + BM25 hybrid"),
    System("minilm+header", MINILM, "data/ablations/minilm_header", False, "MiniLM + context headers"),
    System("minilm+header+bm25", MINILM, "data/ablations/minilm_header", True, "MiniLM + headers + BM25"),
    System("e5", E5, "data/ablations/e5_noheader", False, "e5-large-instruct, vector only"),
    System("e5+bm25", E5, "data/ablations/e5_noheader", True, "e5 + BM25 hybrid"),
    System("e5+header", E5, "data/chroma_db", False, "e5 + context headers (shipped)"),
    System("e5+header+bm25", E5, "data/chroma_db", True, "e5 + headers + BM25"),
]
# The index exactly as it was in git before this work (MiniLM, overlap 250, no text cleaning).
# Its chunk ids are not comparable with the others, so it is only used for the answer evaluation.
# Recreate with: git archive 9cfa18c data/chroma_db | tar -x -C tmp && move to data/ablations/v1_original
V1 = System("v1", MINILM, "data/ablations/v1_original", False, "original index from git 9cfa18c")

BY_NAME = {s.name: s for s in SYSTEMS + [V1]}
BASELINE = "minilm"
SHIPPED = "e5+header"


def activate(system: System):
    """Point the app's settings at this system's index and model, and return the collection."""
    if settings.embedding_model != system.embedding_model:
        settings.embedding_model = system.embedding_model
        vector_store._embedder = None
    settings.chroma_path = system.chroma_path
    vector_store._bm25_indexes.clear()
    return vector_store.load_vector_store()


def retrieve(system: System, collection, question: str, k: int) -> list[dict]:
    return vector_store.search(question, collection, n_results=k, use_hybrid=system.hybrid, use_reranker=False)
