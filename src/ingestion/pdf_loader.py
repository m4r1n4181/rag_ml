import fitz  # pymupdf
import pdfplumber
from pathlib import Path
from loguru import logger
from dataclasses import dataclass, field
from typing import Literal


IMAGE_DIR = Path("data/slide_images")


@dataclass
class Document:
    content: str
    metadata: dict = field(default_factory=dict)


def load_textbook(pdf_path: Path) -> list[Document]:
    """Loads PDF textbook into a list of Documents"""
    documents = []

    with fitz.open(pdf_path) as pdf:
        logger.info(f"Loading textbook: {pdf_path.name} ({len(pdf)} pages)")

        for page_num, page in enumerate(pdf):
            text = page.get_text("text").strip()

            if len(text) < 50:  # skip empty pages
                continue

            documents.append(Document(
                content=text,
                metadata={
                    "source": pdf_path.name,
                    "page": page_num + 1,
                    "type": "textbook",
                    "total_pages": len(pdf)
                }
            ))

    logger.success(f"Loaded {len(documents)} pages from {pdf_path.name}")
    return documents


def load_presentation(pdf_path: Path) -> list[Document]:
    documents = []

    with pdfplumber.open(pdf_path) as pdf:
        logger.info(f"Loading presentation: {pdf_path.name} ({len(pdf.pages)} slides)")

        for slide_num, page in enumerate(pdf.pages):
            text = page.extract_text()

            if not text or len(text.strip()) < 20:
                continue

            lines = [l.strip() for l in text.strip().split('\n') if l.strip()]

            if len(lines) < 3:
                logger.debug(f"Skipping slide {slide_num + 1} (only title): {lines}")
                continue

            title = lines[0] if lines else ""
            is_cover_slide = (
                slide_num == 0 and
                len(text.strip()) < 300
            )
            if is_cover_slide:
                logger.debug(f"Skipping title slide: {pdf_path.name}")
                continue

            # Path to slide pic (important for formulas)
            image_name = f"{pdf_path.stem}_p{slide_num + 1}.png"
            image_path = str(IMAGE_DIR / image_name)

            documents.append(Document(
                content=text.strip(),
                metadata={
                    "source": pdf_path.name,
                    "slide": slide_num + 1,
                    "title": title,
                    "type": "presentation",
                    "total_slides": len(pdf.pages),
                    "image_path": image_path,
                }
            ))

    logger.success(f"Loaded {len(documents)} slides from {pdf_path.name}")
    return documents


def detect_and_load(pdf_path: Path) -> list[Document]:
    """Detecting type of pdf and loading it"""
    with fitz.open(pdf_path) as pdf:
        total_pages = len(pdf)

        # Presentations have less text than textbooks
        sample_pages = min(5, total_pages)
        avg_text_length = sum(
            len(pdf[i].get_text("text"))
            for i in range(sample_pages)
        ) / sample_pages

        if total_pages > 50 and avg_text_length > 500:
            return load_textbook(pdf_path)
        else:
            return load_presentation(pdf_path)