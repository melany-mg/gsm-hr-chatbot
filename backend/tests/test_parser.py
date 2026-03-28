from unittest.mock import MagicMock, patch
from pathlib import Path
from app.parser import extract_from_pdf, extract_from_docx


def _make_mock_pdf(pages):
    """pages: list of (text, tables) tuples"""
    mock_pdf = MagicMock()
    mock_pages = []
    for text, tables in pages:
        page = MagicMock()
        page.extract_text.return_value = text
        page.extract_tables.return_value = tables
        mock_pages.append(page)
    mock_pdf.__enter__ = lambda s: mock_pdf
    mock_pdf.__exit__ = MagicMock(return_value=False)
    mock_pdf.pages = mock_pages
    return mock_pdf


def test_extract_pdf_plain_text():
    mock_pdf = _make_mock_pdf([("Employee benefits overview", [])])
    with patch("app.parser.pdfplumber.open", return_value=mock_pdf):
        results = list(extract_from_pdf(Path("handbook.pdf")))
    assert len(results) == 1
    assert results[0]["text"] == "Employee benefits overview"
    assert results[0]["page"] == 1
    assert results[0]["source"] == "handbook.pdf"


def test_extract_pdf_table():
    table = [["Plan", "Deductible"], ["HMO", "$500"]]
    mock_pdf = _make_mock_pdf([("", [table])])
    with patch("app.parser.pdfplumber.open", return_value=mock_pdf):
        results = list(extract_from_pdf(Path("benefits.pdf")))
    assert len(results) == 1
    assert "Plan | Deductible" in results[0]["text"]
    assert "HMO | $500" in results[0]["text"]


def test_extract_pdf_text_and_table_combined():
    table = [["Plan", "Cost"], ["PPO", "$800"]]
    mock_pdf = _make_mock_pdf([("Overview of plans", [table])])
    with patch("app.parser.pdfplumber.open", return_value=mock_pdf):
        results = list(extract_from_pdf(Path("guide.pdf")))
    assert "Overview of plans" in results[0]["text"]
    assert "PPO | $800" in results[0]["text"]


def test_extract_pdf_skips_empty_pages():
    mock_pdf = _make_mock_pdf([("   ", []), ("Real content", [])])
    with patch("app.parser.pdfplumber.open", return_value=mock_pdf):
        results = list(extract_from_pdf(Path("test.pdf")))
    assert len(results) == 1
    assert results[0]["page"] == 2


def test_extract_pdf_handles_none_cells():
    table = [["Plan", None], ["HMO", "$500"]]
    mock_pdf = _make_mock_pdf([("", [table])])
    with patch("app.parser.pdfplumber.open", return_value=mock_pdf):
        results = list(extract_from_pdf(Path("test.pdf")))
    assert "Plan | " in results[0]["text"]


def test_extract_docx_paragraphs():
    mock_para = MagicMock()
    mock_para.text = "Vacation policy allows 10 days."
    mock_doc = MagicMock()
    mock_doc.paragraphs = [mock_para]
    mock_doc.tables = []
    with patch("app.parser.Document", return_value=mock_doc):
        results = list(extract_from_docx(Path("policy.docx")))
    assert len(results) == 1
    assert "Vacation policy allows 10 days." in results[0]["text"]
    assert results[0]["page"] == 1
    assert results[0]["source"] == "policy.docx"


def test_extract_docx_table():
    mock_cell1 = MagicMock()
    mock_cell1.text = "Plan"
    mock_cell2 = MagicMock()
    mock_cell2.text = "Cost"
    mock_row = MagicMock()
    mock_row.cells = [mock_cell1, mock_cell2]
    mock_table = MagicMock()
    mock_table.rows = [mock_row]
    mock_doc = MagicMock()
    mock_doc.paragraphs = []
    mock_doc.tables = [mock_table]
    with patch("app.parser.Document", return_value=mock_doc):
        results = list(extract_from_docx(Path("policy.docx")))
    assert "Plan | Cost" in results[0]["text"]


def test_extract_docx_skips_empty_paragraphs():
    mock_para1 = MagicMock()
    mock_para1.text = ""
    mock_para2 = MagicMock()
    mock_para2.text = "Real content"
    mock_doc = MagicMock()
    mock_doc.paragraphs = [mock_para1, mock_para2]
    mock_doc.tables = []
    with patch("app.parser.Document", return_value=mock_doc):
        results = list(extract_from_docx(Path("policy.docx")))
    assert results[0]["text"] == "Real content"
