"""
Build the vector store from the PDFs in data/raw.

Run again after changing the embedding model, chunking or the PDFs:
    python build_db.py
"""
import argparse

from src.config import settings
from src.ingestion.ingest import run_ingestion
from src.retrieval.vector_store import build_vector_store

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection", default=settings.collection_name)
    args = parser.parse_args()

    build_vector_store(run_ingestion(), collection_name=args.collection)
