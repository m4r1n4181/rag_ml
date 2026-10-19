from pathlib import Path

from loguru import logger
from tqdm import tqdm

from src.ingestion.chunker import chunk_documents
from src.ingestion.pdf_loader import Document, detect_and_load


def run_ingestion(data_dir: str = "data/raw") -> list[Document]:
    """Load every PDF in data_dir and split it into chunks."""
    pdf_files = sorted(Path(data_dir).glob("*.pdf"))
    if not pdf_files:
        logger.error(f"No PDF files in {data_dir}")
        return []

    documents = []
    for pdf_path in tqdm(pdf_files, desc="Loading PDFs"):
        try:
            documents.extend(detect_and_load(pdf_path))
        except Exception as e:
            logger.error(f"Could not read {pdf_path.name}: {e}")

    chunks = chunk_documents(documents)
    logger.success(f"Ingestion finished: {len(chunks)} chunks from {len(pdf_files)} files")
    return chunks


if __name__ == "__main__":
    chunks = run_ingestion()
    textbook = sum(c.metadata.get("type") == "textbook" for c in chunks)
    print(f"Textbook chunks: {textbook}\nSlide chunks:    {len(chunks) - textbook}\nTotal:           {len(chunks)}")
