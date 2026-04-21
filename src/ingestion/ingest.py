import sys
from pathlib import Path
from loguru import logger
from tqdm import tqdm # loading bar

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.ingestion.pdf_loader import detect_and_load
from src.ingestion.chunker import chunk_documents


def run_ingestion(data_dir: str = "data/raw") -> list:
    """Loads all PDFs from the raw data directory and chunks them"""
    raw_dir = Path(data_dir)
    pdf_files = list(raw_dir.glob("*.pdf"))

    if not pdf_files:
        logger.error(f"No PDF files in {raw_dir}")
        return []

    logger.info(f"Found {len(pdf_files)} PDF files")

    all_documents = []
    for pdf_path in tqdm(pdf_files, desc="Loading PDFs"):
        try:
            docs = detect_and_load(pdf_path)
            all_documents.extend(docs)
        except Exception as e:
            logger.error(f"Error while reading {pdf_path.name}: {e}")

    chunks = chunk_documents(all_documents)

    logger.success(f"Ingestion finished: {len(chunks)} chunks total")
    return chunks


if __name__ == "__main__":
    chunks = run_ingestion()

    # statistics
    textbook_chunks = [c for c in chunks if c.metadata.get("type") == "textbook"]
    presentation_chunks = [c for c in chunks if c.metadata.get("type") == "presentation"]

    print(f"\n{'=' * 40}")
    print(f"Textbook:       {len(textbook_chunks)} chunks")
    print(f"Presentations: {len(presentation_chunks)} chunks")
    print(f"Tptal:       {len(chunks)} chunks")
    print(f"{'=' * 40}")