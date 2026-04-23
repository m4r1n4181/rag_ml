import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer, CrossEncoder
from loguru import logger
from tqdm import tqdm
from src.config import settings

from src.ingestion.pdf_loader import Document

BATCH_SIZE = settings.batch_size
CHROMA_PATH = settings.chroma_path
COLLECTION_NAME = settings.collection_name

_embedder = None
_reranker = None


def get_embedder() -> SentenceTransformer:
    """
    Singleton pattern - loading model only once
    = better performance
    """
    global _embedder
    if _embedder is None:
        logger.info("Loading embedding model...")
        _embedder = SentenceTransformer(settings.embedding_model) #for Serbian
        logger.success("Embedding model ready.")
    return _embedder


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    List of text -> list of embeddings (vectors)

    normalize_embeddings=True -> length = 1
    ChromaDB uses cosine similarity,
    and for eigen vectors cosine = dot product -> faster.
    """
    embedder = get_embedder()
    vectors = embedder.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True
    )
    return vectors.tolist()


def embed_query(query: str) -> list[float]:
    """
    Embedding query
    for later upgrades
    """
    embedder = get_embedder()
    vector = embedder.encode(
        query,
        normalize_embeddings=True,
        convert_to_numpy=True
    )
    return vector.tolist()

def get_reranker() -> CrossEncoder:
    """
    Lazy-load cross-encoder reranker.
    """
    global _reranker
    if _reranker is None:
        logger.info("Loading cross-encoder reranker...")
        _reranker = CrossEncoder(settings.reranker_model)
        logger.success("Reranker model ready.")
    return _reranker
def rerank(query: str, hits: list[dict], top_n: int = 5) -> list[dict]:
    """
    Rerank top hits using a cross-encoder.
    hits: list[{"content": str, "metadata": dict, "similarity": float}]
    Returns same format, but better sorted.
    """
    if not hits:
        return hits

    reranker = get_reranker()

    # getting pairs ready (query, document_text)
    pairs = [(query, h["content"]) for h in hits]

    # CrossEncoder gives list of scores
    scores = reranker.predict(pairs)

    # enters scores into hits and sorts them
    for h, s in zip(hits, scores):
        h["rerank_score"] = float(s)

    hits.sort(key=lambda x: x["rerank_score"], reverse=True)

    return hits[:top_n]

def get_chroma_client():
    return chromadb.PersistentClient(
        path=CHROMA_PATH,
        settings=Settings(anonymized_telemetry=False)
    )

def build_vector_store(chunks: list[Document], force_rebuild: bool = False):
    client = get_chroma_client()
    existing = [c.name for c in client.list_collections()]

    if COLLECTION_NAME in existing:
        if force_rebuild:
            logger.warning("Deleting old collection and building new...")
            client.delete_collection(COLLECTION_NAME)
        else:
            logger.info("Vector store already exists. Using existing.")
            return client.get_collection(COLLECTION_NAME)

    logger.info(f"Building vector store of {len(chunks)} chunks...")
    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    for i in tqdm(range(0, len(chunks), BATCH_SIZE), desc="Writing in ChromaDB"):
        batch = chunks[i:i + BATCH_SIZE]

        texts = [doc.content for doc in batch]
        metadatas = [doc.metadata for doc in batch]
        ids = [f"chunk_{i + j}" for j in range(len(batch))]

        # embed_texts already uses batches, but batching ChromaDB entries too
        embeddings = embed_texts(texts)

        collection.add(
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )

    logger.success(f"Vector store built: {collection.count()} vectors")
    return collection


def load_vector_store():
    client = get_chroma_client()
    existing = [c.name for c in client.list_collections()]

    if COLLECTION_NAME not in existing:
        raise RuntimeError(
            "Vector store doesn't exist. Run build_db.py."
        )

    collection = client.get_collection(COLLECTION_NAME)
    logger.info(f"Loaded vector store: {collection.count()} vectors")
    return collection


def search(query: str, collection, n_results: int = 5) -> list[dict]:
    """
    Semantic search on vector store + cross-encoder reranking

    """
    n_results_query = max(n_results * 3, n_results + 5) #for reranking, more candidates from Chroma
    query_vector = embed_query(query)

    results = collection.query(
        query_embeddings=[query_vector],
        n_results=n_results_query,
        include=["documents", "metadatas", "distances"]
    )

    hits = []
    for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
    ):
        hits.append({
            "content": doc,
            "metadata": meta,
            # cosine distance → similarity
            # dist=0.0 identical, dist=2.0 opposite
            "similarity": round(1 - dist, 3)
        })

        # # 2) reranking
        # try:
        #     hits = rerank(query, hits, top_n=n_results)
        # except Exception as e:
        #     logger.error(f"Reranker failed, falling back to pure vector ranking: {e}")
        #     # fallback
        hits = sorted(hits, key=lambda x: x["similarity"], reverse=True)[:n_results]

    return hits
