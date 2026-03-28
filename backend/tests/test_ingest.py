from unittest.mock import MagicMock, patch, call
from pathlib import Path
import pytest


def test_ingest_wipes_existing_collection(tmp_path):
    (tmp_path / "test.pdf").write_bytes(b"fake")

    mock_client = MagicMock()
    mock_client.collection_exists.return_value = True
    mock_model = MagicMock()
    mock_model.encode.return_value = [0.1] * 768

    with patch("app.ingest.DOCUMENTS_DIR", tmp_path), \
         patch("app.ingest.QdrantClient", return_value=mock_client), \
         patch("app.ingest.SentenceTransformer", return_value=mock_model), \
         patch("app.ingest.extract_from_pdf", return_value=[
             {"text": "Hello", "page": 1, "source": "test.pdf"}
         ]):
        from app.ingest import run_ingest
        run_ingest()

    mock_client.delete_collection.assert_called_once()
    mock_client.create_collection.assert_called_once()


def test_ingest_skips_delete_when_collection_absent(tmp_path):
    (tmp_path / "test.pdf").write_bytes(b"fake")

    mock_client = MagicMock()
    mock_client.collection_exists.return_value = False
    mock_model = MagicMock()
    mock_model.encode.return_value = [0.1] * 768

    with patch("app.ingest.DOCUMENTS_DIR", tmp_path), \
         patch("app.ingest.QdrantClient", return_value=mock_client), \
         patch("app.ingest.SentenceTransformer", return_value=mock_model), \
         patch("app.ingest.extract_from_pdf", return_value=[
             {"text": "Hello", "page": 1, "source": "test.pdf"}
         ]):
        from app.ingest import run_ingest
        run_ingest()

    mock_client.delete_collection.assert_not_called()
    mock_client.create_collection.assert_called_once()


def test_ingest_upserts_points(tmp_path):
    (tmp_path / "test.pdf").write_bytes(b"fake")

    mock_client = MagicMock()
    mock_client.collection_exists.return_value = False
    mock_model = MagicMock()
    mock_model.encode.return_value = [0.0] * 768

    with patch("app.ingest.DOCUMENTS_DIR", tmp_path), \
         patch("app.ingest.QdrantClient", return_value=mock_client), \
         patch("app.ingest.SentenceTransformer", return_value=mock_model), \
         patch("app.ingest.extract_from_pdf", return_value=[
             {"text": "Benefits info", "page": 2, "source": "test.pdf"}
         ]):
        from app.ingest import run_ingest
        run_ingest()

    mock_client.upsert.assert_called_once()
    upsert_call = mock_client.upsert.call_args
    points = upsert_call.kwargs["points"]
    assert len(points) >= 1
    assert points[0].payload["source"] == "test.pdf"
    assert points[0].payload["page"] == 2


def test_ingest_skips_unsupported_files(tmp_path):
    (tmp_path / "notes.txt").write_text("ignore me")

    mock_client = MagicMock()
    mock_client.collection_exists.return_value = False
    mock_model = MagicMock()

    with patch("app.ingest.DOCUMENTS_DIR", tmp_path), \
         patch("app.ingest.QdrantClient", return_value=mock_client), \
         patch("app.ingest.SentenceTransformer", return_value=mock_model):
        from app.ingest import run_ingest
        run_ingest()

    mock_client.upsert.assert_not_called()


def test_ingest_processes_docx(tmp_path):
    (tmp_path / "policy.docx").write_bytes(b"fake")

    mock_client = MagicMock()
    mock_client.collection_exists.return_value = False
    mock_model = MagicMock()
    mock_model.encode.return_value = [0.0] * 768

    with patch("app.ingest.DOCUMENTS_DIR", tmp_path), \
         patch("app.ingest.QdrantClient", return_value=mock_client), \
         patch("app.ingest.SentenceTransformer", return_value=mock_model), \
         patch("app.ingest.extract_from_docx", return_value=[
             {"text": "Policy content", "page": 1, "source": "policy.docx"}
         ]):
        from app.ingest import run_ingest
        run_ingest()

    mock_client.upsert.assert_called_once()
