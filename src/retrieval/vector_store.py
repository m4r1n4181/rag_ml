import chromadb
from chromadb.config import Settings
from loguru import logger
from sentence_transformers import CrossEncoder, SentenceTransformer
from tqdm import tqdm

from src.config import settings
from src.ingestion.pdf_loader import Document
from src.retrieval.lexical import BM25Index

_embedder = None
_reranker = None
_bm25_indexes: dict[str, BM25Index] = {}


# --- Embeddings ------------------------------------------------------------

def get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        logger.info(f"Loading embedding model {settings.embedding_model}")
        kwargs = {}
        if settings.embedding_dtype != "float32":
            import torch
            kwargs["torch_dtype"] = getattr(torch, settings.embedding_dtype)
        # model_kwargs only when needed, so float32 also works with older sentence-transformers
        _embedder = SentenceTransformer(settings.embedding_model, **({"model_kwargs": kwargs} if kwargs else {}))
    return _embedder


def uses_query_instruction() -> bool:
    name = settings.embedding_model.lower()
    return "e5" in name and "instruct" in name


def embed_texts(texts: list[str]) -> list[list[float]]:
    vectors = get_embedder().encode(
        texts, batch_size=settings.batch_size, normalize_embeddings=True, convert_to_numpy=True
    )
    return vectors.tolist()


def embed_query(query: str) -> list[float]:
    # e5-instruct is asymmetric: queries get a task instruction, documents don't.
    if uses_query_instruction():
        query = f"Instruct: {settings.query_instruction}\nQuery: {query}"
    return get_embedder().encode(query, normalize_embeddings=True, convert_to_numpy=True).tolist()


# --- Chroma ----------------------------------------------------------------

def get_chroma_client():
    return chromadb.PersistentClient(path=settings.chroma_path, settings=Settings(anonymized_telemetry=False))


def chunk_id(doc: Document) -> str:
    m = doc.metadata
    if m.get("type") == "presentation":
        return f"{m.get('source')}::s{m['slide']}"
    return f"{m.get('source')}::p{m.get('page')}_c{m.get('chunk', 1)}"


def build_vector_store(chunks: list[Document], collection_name: str | None = None):
    """
    Build into a temporary collection and swap it in only when it is complete,
    so a crash halfway through never leaves the app without an index.
    """
    name = collection_name or settings.collection_name
    tmp_name = f"{name}__building"
    client = get_chroma_client()
    existing = {c.name for c in client.list_collections()}
    if tmp_name in existing:
        client.delete_collection(tmp_name)

    collection = client.create_collection(
        name=tmp_name, metadata={"hnsw:space": "cosine", "embedding_model": settings.embedding_model}
    )
    for i in tqdm(range(0, len(chunks), settings.batch_size), desc="Embedding"):
        batch = chunks[i:i + settings.batch_size]
        collection.add(
            ids=[chunk_id(d) for d in batch],
            embeddings=embed_texts([d.content for d in batch]),
            documents=[d.content for d in batch],
            metadatas=[d.metadata for d in batch],
        )

    if name in existing:
        client.delete_collection(name)
    collection.modify(name=name)
    logger.success(f"Vector store '{name}' built: {collection.count()} chunks")
    return client.get_collection(name)


def load_vector_store(collection_name: str | None = None):
    name = collection_name or settings.collection_name
    client = get_chroma_client()
    if name not in {c.name for c in client.list_collections()}:
        raise RuntimeError(f"Collection '{name}' not found in {settings.chroma_path}. Run build_db.py first.")

    collection = client.get_collection(name)
    built_with = (collection.metadata or {}).get("embedding_model")
    if built_with and built_with != settings.embedding_model:
        raise RuntimeError(
            f"Index was built with '{built_with}' but EMBEDDING_MODEL is '{settings.embedding_model}'. "
            "Rebuild with build_db.py or fix .env."
        )
    return collection


# --- Search ----------------------------------------------------------------

def vector_search(query: str, collection, n: int) -> list[dict]:
    res = collection.query(
        query_embeddings=[embed_query(query)], n_results=n, include=["documents", "metadatas", "distances"]
    )
    return [
        {"id": i, "content": doc, "metadata": meta, "similarity": round(1 - dist, 3)}
        for i, doc, meta, dist in zip(res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0])
    ]


def bm25_search(query: str, collection, n: int) -> list[dict]:
    if collection.name not in _bm25_indexes:
        data = collection.get(include=["documents", "metadatas"])
        _bm25_indexes[collection.name] = BM25Index(data["ids"], data["documents"], data["metadatas"])
    return _bm25_indexes[collection.name].search(query, n)


def reciprocal_rank_fusion(result_lists: list[list[dict]], k: int = 60) -> list[dict]:
    """score(d) = sum over lists of 1 / (k + rank of d). Uses ranks only, so score scales don't matter."""
    fused: dict[str, dict] = {}
    for results in result_lists:
        for rank, hit in enumerate(results, start=1):
            entry = fused.setdefault(hit["id"], {**hit, "rrf": 0.0})
            entry["rrf"] += 1.0 / (k + rank)
            if "similarity" in hit:
                entry["similarity"] = hit["similarity"]
    return sorted(fused.values(), key=lambda h: h["rrf"], reverse=True)


def get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        logger.info(f"Loading reranker {settings.reranker_model}")
        _reranker = CrossEncoder(settings.reranker_model, max_length=512)
    return _reranker


def rerank(query: str, hits: list[dict], top_n: int) -> list[dict]:
    scores = get_reranker().predict([(query, h["content"]) for h in hits])
    for h, s in zip(hits, scores):
        h["rerank_score"] = float(s)
    return sorted(hits, key=lambda h: h["rerank_score"], reverse=True)[:top_n]


def search(
    query: str,
    collection,
    n_results: int | None = None,
    use_hybrid: bool | None = None,
    use_reranker: bool | None = None,
) -> list[dict]:
    """Vector search, optionally fused with BM25 (RRF) and reranked with a cross-encoder."""
    n_results = n_results or settings.n_results
    use_hybrid = settings.use_hybrid if use_hybrid is None else use_hybrid
    use_reranker = settings.use_reranker if use_reranker is None else use_reranker

    n_candidates = max(n_results * 4, 20)
    hits = vector_search(query, collection, n_candidates)
    if use_hybrid:
        hits = reciprocal_rank_fusion([hits, bm25_search(query, collection, n_candidates)])

    if use_reranker and hits:
        try:
            return rerank(query, hits, n_results)
        except Exception as e:
            logger.error(f"Reranker failed, using unreranked results: {e}")

    for h in hits:
        h.setdefault("similarity", 0.0)
    return hits[:n_results]
