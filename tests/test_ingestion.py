from pathlib import Path

from src.ingestion.chunker import context_header, with_header
from src.ingestion.pdf_loader import clean_slide_text, clean_textbook_text, lecture_name


def test_lecture_name_strips_lecture_number():
    assert lecture_name(Path("L02 Linearna regresija.pdf")) == "Linearna regresija"
    assert lecture_name(Path("L02_1 Linearna regresija_uticaj.pdf")) == "Linearna regresija uticaj"
    assert lecture_name(Path("L07 - KDE.pdf")) == "KDE"
    assert lecture_name(Path("MU_praktikum.pdf")) == "MU praktikum"


def test_clean_slide_text_removes_footer_and_bullets():
    raw = "Lasso\n Alternativa ridge regresiji\n\nMašinsko učenje 1\nKomunikacione tehnologije i obrada signala 29"
    assert clean_slide_text(raw) == "Lasso\nAlternativa ridge regresiji"


def test_clean_slide_text_handles_glued_footer():
    assert clean_slide_text("Lasso rezultuje retkim modelom\nMašinskoučenje1") == "Lasso rezultuje retkim modelom"


def test_context_header_for_slide_and_textbook():
    slide = {"type": "presentation", "lecture": "Linearna regresija", "title": "Lasso"}
    page = {"type": "textbook", "lecture": "MU praktikum", "page": 45}
    assert context_header(slide) == "Predavanje: Linearna regresija | Slajd: Lasso"
    assert context_header(page) == "Praktikum: MU praktikum | Strana 45"
    assert with_header(slide, "tekst").startswith("Predavanje: Linearna regresija")


def test_clean_textbook_text_restores_serbian_letters():
    raw = "regularizacione tehnike ˇciji je cilj ograniˇcenje koeﬁcijenata izmedu pristrasnosti, pove´cava"
    assert clean_textbook_text(raw) == (
        "regularizacione tehnike čiji je cilj ograničenje koeficijenata između pristrasnosti, povećava"
    )
