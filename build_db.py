"""
Run only once to build vector store
"""
from src.ingestion.ingest import run_ingestion
from src.retrieval.vector_store import build_vector_store

if __name__ == "__main__":
    chunks = run_ingestion()
    build_vector_store(chunks, force_rebuild=True)