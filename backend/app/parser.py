import pdfplumber
from docx import Document
from pathlib import Path
from typing import Generator


def extract_from_pdf(path: Path) -> Generator[dict, None, None]:
    """Yield one dict per non-empty page: {text, page, source}. Tables are
    serialized as pipe-delimited rows so the LLM can reason over them."""
    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            parts = []
            text = page.extract_text() or ""
            if text.strip():
                parts.append(text.strip())
            for table in page.extract_tables():
                rows = []
                for row in table:
                    cells = [cell if cell is not None else "" for cell in row]
                    rows.append(" | ".join(cells))
                parts.append("\n".join(rows))
            combined = "\n".join(parts).strip()
            if combined:
                yield {"text": combined, "page": i, "source": path.name}


def extract_from_docx(path: Path) -> Generator[dict, None, None]:
    """Yield a single dict for the whole document: {text, page, source}.
    DOCX files have no native page numbers so page is always 1."""
    doc = Document(str(path))
    parts = []
    for para in doc.paragraphs:
        if para.text.strip():
            parts.append(para.text.strip())
    for table in doc.tables:
        rows = []
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells]
            rows.append(" | ".join(cells))
        parts.append("\n".join(rows))
    combined = "\n".join(parts).strip()
    if combined:
        yield {"text": combined, "page": 1, "source": path.name}
