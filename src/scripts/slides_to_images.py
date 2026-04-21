# scripts/export_slide_images.py
from pathlib import Path
import fitz  # PyMuPDF

PDF_DIR = Path("data/raw")
OUT_DIR = Path("data/slide_images")
OUT_DIR.mkdir(parents=True, exist_ok=True)

for pdf_path in PDF_DIR.glob("*.pdf"):
    doc = fitz.open(pdf_path)
    for page_index in range(len(doc)):
        page = doc[page_index]
        pix = page.get_pixmap(dpi=150)
        out_name = f"{pdf_path.stem}_p{page_index + 1}.png"
        out_path = OUT_DIR / out_name
        pix.save(out_path)
    doc.close()