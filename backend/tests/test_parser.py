from unittest.mock import MagicMock, patch, PropertyMock
from pathlib import Path
from app.parser import extract_from_pdf, extract_from_docx


def _make_mock_pdf(pages, table_bboxes_per_page=None):
    """pages: list of (text, tables) tuples.
    table_bboxes_per_page: optional list of bbox lists per page; if omitted,
    find_tables() returns empty lists so no filtering is applied.
    """
    if table_bboxes_per_page is None:
        table_bboxes_per_page = [[] for _ in pages]

    mock_pdf = MagicMock()
    mock_pages = []
    for (text, tables), bboxes in zip(pages, table_bboxes_per_page):
        page = MagicMock()
        page.extract_text.return_value = text

        # find_tables() returns objects with a .bbox attribute
        mock_table_finders = []
        for bbox in bboxes:
            tf = MagicMock()
            tf.bbox = bbox
            mock_table_finders.append(tf)
        page.find_tables.return_value = mock_table_finders

        # filter() returns a filtered page whose extract_text returns the
        # same text (in unit tests we're not doing real spatial filtering)
        filtered_page = MagicMock()
        filtered_page.extract_text.return_value = text
        page.filter.return_value = filtered_page

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
    """Table rows are serialized with column-header context."""
    table = [["Plan", "Deductible"], ["HMO", "$500"]]
    mock_pdf = _make_mock_pdf([("", [table])])
    with patch("app.parser.pdfplumber.open", return_value=mock_pdf):
        results = list(extract_from_pdf(Path("benefits.pdf")))
    assert len(results) == 1
    text = results[0]["text"]
    # Header row is preserved as pipe-delimited
    assert "Plan | Deductible" in text
    # Data row appears as key:value pairs
    assert "Plan: HMO" in text
    assert "Deductible: $500" in text


def test_extract_pdf_text_and_table_combined():
    table = [["Plan", "Cost"], ["PPO", "$800"]]
    mock_pdf = _make_mock_pdf([("Overview of plans", [table])])
    with patch("app.parser.pdfplumber.open", return_value=mock_pdf):
        results = list(extract_from_pdf(Path("guide.pdf")))
    text = results[0]["text"]
    assert "Overview of plans" in text
    assert "Plan: PPO" in text
    assert "Cost: $800" in text


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
    text = results[0]["text"]
    assert "Plan: HMO" in text


def test_extract_pdf_no_double_extraction():
    """When a page has tables, table cell text must not appear in the prose
    section.  We simulate this by giving the *filtered* page a text output
    that excludes the table cell content ('HMO', '$500'), while the
    *unfiltered* page.extract_text() would have included it."""
    table = [["Plan", "Deductible"], ["HMO", "$500"]]

    mock_pdf = MagicMock()
    page = MagicMock()

    # Unfiltered text includes table cell content (pdfplumber default behaviour)
    page.extract_text.return_value = "Benefits overview\nHMO $500"

    # find_tables() signals that a table region exists
    tf = MagicMock()
    tf.bbox = (50, 100, 400, 200)
    page.find_tables.return_value = [tf]

    # filter() returns a page whose extract_text strips the table region
    filtered_page = MagicMock()
    filtered_page.extract_text.return_value = "Benefits overview"
    page.filter.return_value = filtered_page

    page.extract_tables.return_value = [table]

    mock_pdf.__enter__ = lambda s: mock_pdf
    mock_pdf.__exit__ = MagicMock(return_value=False)
    mock_pdf.pages = [page]

    with patch("app.parser.pdfplumber.open", return_value=mock_pdf):
        results = list(extract_from_pdf(Path("test.pdf")))

    text = results[0]["text"]

    # Prose section comes from the *filtered* page — table cells not duplicated
    assert "Benefits overview" in text
    # Table content appears exactly once, via the structured serialization
    assert text.count("HMO") == 1
    assert text.count("$500") == 1
    # The filter was actually applied (not the raw extract_text)
    page.filter.assert_called_once()


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
    """DOCX table rows also use key:value serialization."""
    mock_cell1 = MagicMock()
    mock_cell1.text = "Plan"
    mock_cell2 = MagicMock()
    mock_cell2.text = "Cost"
    mock_cell3 = MagicMock()
    mock_cell3.text = "PPO"
    mock_cell4 = MagicMock()
    mock_cell4.text = "$800"
    mock_row1 = MagicMock()
    mock_row1.cells = [mock_cell1, mock_cell2]
    mock_row2 = MagicMock()
    mock_row2.cells = [mock_cell3, mock_cell4]
    mock_table = MagicMock()
    mock_table.rows = [mock_row1, mock_row2]
    mock_doc = MagicMock()
    mock_doc.paragraphs = []
    mock_doc.tables = [mock_table]
    with patch("app.parser.Document", return_value=mock_doc):
        results = list(extract_from_docx(Path("policy.docx")))
    text = results[0]["text"]
    assert "Plan | Cost" in text
    assert "Plan: PPO" in text
    assert "Cost: $800" in text


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
