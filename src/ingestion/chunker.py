from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger

from src.config import settings
from src.ingestion.pdf_loader import Document


def context_header(metadata: dict) -> str:
    """
    One line of context prepended to every chunk before embedding.

    A slide titled "Lasso" never says "regularizacija" or "linearna regresija",
    so without the lecture name it is hard to match to a question about
    regularization in linear regression.
    """
    if not settings.use_context_header:
        return ""
    lecture = metadata.get("lecture", metadata.get("source", ""))
    if metadata.get("type") == "presentation":
        return f"Predavanje: {lecture} | Slajd: {metadata.get('title', '')}"
    return f"Praktikum: {lecture} | Strana {metadata.get('page', '?')}"


def with_header(metadata: dict, text: str) -> str:
    header = context_header(metadata)
    return f"{header}\n{text}" if header else text


def chunk_textbook(documents: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " "],
    )

    chunks = []
    for doc in documents:
        splits = splitter.split_text(doc.content)
        for i, split in enumerate(splits):
            meta = {**doc.metadata, "chunk": i + 1, "total_chunks_on_page": len(splits)}
            chunks.append(Document(content=with_header(meta, split), metadata=meta))

    logger.info(f"Textbook: {len(documents)} pages -> {len(chunks)} chunks")
    return chunks


def chunk_presentation(documents: list[Document]) -> list[Document]:
    """Slides are short and self-contained, so one slide is one chunk."""
    chunks = [Document(content=with_header(d.metadata, d.content), metadata=d.metadata) for d in documents]
    logger.info(f"Presentations: {len(chunks)} slides")
    return chunks


def chunk_documents(documents: list[Document]) -> list[Document]:
    textbook = [d for d in documents if d.metadata.get("type") == "textbook"]
    slides = [d for d in documents if d.metadata.get("type") == "presentation"]
    return chunk_textbook(textbook) + chunk_presentation(slides)
