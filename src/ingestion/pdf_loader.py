import re
from dataclasses import dataclass, field
from pathlib import Path

import fitz  # pymupdf
import pdfplumber
from loguru import logger

from src.config import settings

IMAGE_DIR = Path("data/slide_images")

# Footer printed on every slide of this course
FOOTER_PATTERNS = [
    r"Mašinsko\s*učenje\s*1",
    r"Komunikacione\s*tehnologije\s*i\s*obrada\s*signala\s*\d*",
]
# PowerPoint bullets are extracted as Unicode private-use characters
PRIVATE_USE_CHARS = r"[\ue000-\uf8ff]"

# The textbook is a LaTeX PDF: accents come out as separate characters ("ˇc" instead of "č"),
# ligatures as single glyphs, and "đ" is lost entirely ("izmedu").
TEXTBOOK_FIXES = {
    "ˇc": "č", "ˇC": "Č", "ˇs": "š", "ˇS": "Š", "ˇz": "ž", "ˇZ": "Ž", "´c": "ć", "´C": "Ć",
    "\ufb01": "fi", "\ufb02": "fl", "\ufb03": "ffi", "\ufb00": "ff",
}


@dataclass
class Document:
    content: str
    metadata: dict = field(default_factory=dict)


def lecture_name(pdf_path: Path) -> str:
    """'L02 Linearna regresija.pdf' -> 'Linearna regresija'"""
    name = re.sub(r"^L\d+(_\d+)?\s*-?\s*", "", pdf_path.stem)
    return name.replace("_", " ").strip() or pdf_path.stem


def clean_slide_text(text: str) -> str:
    for pattern in FOOTER_PATTERNS:
        text = re.sub(pattern, "", text)
    text = re.sub(PRIVATE_USE_CHARS, "", text)
    lines = (re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n"))
    return "\n".join(line for line in lines if line)


def clean_textbook_text(text: str) -> str:
    for broken, fixed in TEXTBOOK_FIXES.items():
        text = text.replace(broken, fixed)
    return re.sub(r"\b(iz)?medu", lambda m: (m.group(1) or "") + "među", text)


def load_textbook(pdf_path: Path) -> list[Document]:
    documents = []

    with fitz.open(pdf_path) as pdf:
        logger.info(f"Loading textbook: {pdf_path.name} ({len(pdf)} pages)")

        for page_num, page in enumerate(pdf):
            text = clean_textbook_text(page.get_text("text")).strip()
            if len(text) < 50:
                continue

            documents.append(Document(
                content=text,
                metadata={
                    "source": pdf_path.name,
                    "page": page_num + 1,
                    "type": "textbook",
                    "lecture": lecture_name(pdf_path),
                    "total_pages": len(pdf),
                },
            ))

    logger.success(f"Loaded {len(documents)} pages from {pdf_path.name}")
    return documents


def load_presentation(pdf_path: Path) -> list[Document]:
    documents = []

    with pdfplumber.open(pdf_path) as pdf:
        logger.info(f"Loading presentation: {pdf_path.name} ({len(pdf.pages)} slides)")

        for slide_num, page in enumerate(pdf.pages):
            text = clean_slide_text(page.extract_text() or "")
            lines = [line.strip() for line in text.split("\n") if line.strip()]

            # Title-only slides. Formula slides can have very little text, so the bar is low.
            if len(lines) < 2 or len(text) < 40:
                continue
            # Cover slide
            if slide_num == 0 and len(text) < 300:
                continue

            documents.append(Document(
                content=text,
                metadata={
                    "source": pdf_path.name,
                    "slide": slide_num + 1,
                    "title": lines[0],
                    "type": "presentation",
                    "lecture": lecture_name(pdf_path),
                    "total_slides": len(pdf.pages),
                    "image_path": str(IMAGE_DIR / f"{pdf_path.stem}_p{slide_num + 1}.png"),
                },
            ))

    logger.success(f"Loaded {len(documents)} slides from {pdf_path.name}")
    return documents


def is_textbook(pdf_path: Path) -> bool:
    if settings.textbook_files:
        return pdf_path.name in settings.textbook_files

    # Fallback heuristic when no list is configured: long documents with dense pages
    with fitz.open(pdf_path) as pdf:
        sample = min(5, len(pdf))
        avg_chars = sum(len(pdf[i].get_text("text")) for i in range(sample)) / sample
        return len(pdf) > 50 and avg_chars > 500


def detect_and_load(pdf_path: Path) -> list[Document]:
    if is_textbook(pdf_path):
        return load_textbook(pdf_path)
    return load_presentation(pdf_path)
