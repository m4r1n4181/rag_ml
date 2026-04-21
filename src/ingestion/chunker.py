from langchain.text_splitter import RecursiveCharacterTextSplitter
from loguru import logger
from .pdf_loader import Document
from src.config import settings


def chunk_textbook(documents: list[Document]) -> list[Document]:
    """Split textbook into chunks of 800 characters with overlap of 150"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " "],
        length_function=len
    )

    chunked = []
    for doc in documents:
        splits = splitter.split_text(doc.content)

        for i, split in enumerate(splits):
            chunked.append(Document(
                content=split,
                metadata={
                    **doc.metadata,
                    "chunk": i + 1,
                    "total_chunks_on_page": len(splits)
                }
            ))

    logger.info(f"Book: {len(documents)} pages - {len(chunked)} chunks")
    return chunked


def chunk_presentation(documents: list[Document]) -> list[Document]:
    """One slide = one chunk"""
    logger.info(f"Presentation: {len(documents)} slides")
    return documents


def chunk_documents(documents: list[Document]) -> list[Document]:
    """According to document type, chunk them differently"""
    textbook_docs = [d for d in documents if d.metadata.get("type") == "textbook"]
    presentation_docs = [d for d in documents if d.metadata.get("type") == "presentation"]

    result = []
    if textbook_docs:
        result.extend(chunk_textbook(textbook_docs))
    if presentation_docs:
        result.extend(chunk_presentation(presentation_docs))

    return result