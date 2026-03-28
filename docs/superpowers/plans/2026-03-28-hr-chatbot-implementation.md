# GSM HR Chatbot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a multilingual RAG-based HR chatbot with a Python/FastAPI backend, React frontend, and Qdrant vector database, running in Docker Compose.

**Architecture:** The backend uses LangChain + sentence-transformers for RAG over HR documents stored in Qdrant. The LLM switches between Ollama (dev) and Claude (prod) via environment flag. The frontend is a minimal React chat UI served by Nginx.

**Tech Stack:** Python 3.11, FastAPI, LangChain, sentence-transformers, Qdrant, pdfplumber, python-docx, React 18, Vite, Nginx, Docker Compose

---

## File Map

```
HR Chatbot /
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── config.py          # Pydantic-settings: all env vars
│   │   ├── parser.py          # PDF + DOCX text/table extraction
│   │   ├── ingest.py          # Manual indexing script (run on host)
│   │   ├── chat.py            # RAG retrieval + LLM call + citations
│   │   └── main.py            # FastAPI app: /api/chat, /api/health
│   ├── tests/
│   │   ├── __init__.py
│   │   ├── conftest.py
│   │   ├── test_config.py
│   │   ├── test_parser.py
│   │   ├── test_ingest.py
│   │   ├── test_chat.py
│   │   └── test_main.py
│   ├── documents/
│   │   └── .gitkeep
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   ├── src/
│   │   ├── main.jsx
│   │   ├── App.jsx
│   │   ├── App.css
│   │   ├── api.js
│   │   ├── components/
│   │   │   ├── LanguageDropdown.jsx
│   │   │   ├── MessageBubble.jsx
│   │   │   ├── InputBar.jsx
│   │   │   └── ChatWindow.jsx
│   │   └── __tests__/
│   │       ├── api.test.js
│   │       ├── LanguageDropdown.test.jsx
│   │       ├── MessageBubble.test.jsx
│   │       └── InputBar.test.jsx
│   ├── public/
│   │   └── index.html
│   ├── vite.config.js
│   ├── package.json
│   ├── nginx.conf
│   └── Dockerfile
├── docker-compose.yml
├── .env                  # gitignored
├── .env.example
└── .gitignore
```

---

## Task 1: Project Scaffolding

**Files:**
- Create: `.gitignore`
- Create: `.env.example`
- Create: `.env`
- Create: `backend/documents/.gitkeep`
- Create: `backend/app/__init__.py`
- Create: `backend/tests/__init__.py`

- [ ] **Step 1: Create `.gitignore`**

```
.env
__pycache__/
*.pyc
*.pyo
.venv/
node_modules/
backend/documents/*
!backend/documents/.gitkeep
dist/
.DS_Store
```

- [ ] **Step 2: Create `.env.example`**

```env
ENVIRONMENT=development

# Ollama (dev only)
OLLAMA_BASE_URL=http://host.docker.internal:11434
OLLAMA_MODEL=qwen2.5:7b

# Claude (prod only)
ANTHROPIC_API_KEY=
CLAUDE_MODEL=claude-sonnet-4-6

# Qdrant (used by backend container via Docker networking)
QDRANT_HOST=qdrant
QDRANT_PORT=6333
QDRANT_COLLECTION=hr_documents

# Qdrant (used by ingest script running on host)
QDRANT_EXTERNAL_HOST=localhost

# Embedding (same in dev and prod)
EMBEDDING_MODEL=paraphrase-multilingual-mpnet-base-v2
```

- [ ] **Step 3: Create `.env` (your working local copy)**

Copy `.env.example` to `.env`. Fill in `ANTHROPIC_API_KEY` when deploying to prod. Leave all other values as-is for local development.

- [ ] **Step 4: Create empty init files and gitkeep**

```bash
touch "backend/app/__init__.py"
touch "backend/tests/__init__.py"
touch "backend/documents/.gitkeep"
```

- [ ] **Step 5: Commit**

```bash
git add .gitignore .env.example backend/app/__init__.py backend/tests/__init__.py backend/documents/.gitkeep
git commit -m "chore: project scaffolding"
```

---

## Task 2: Backend Config

**Files:**
- Create: `backend/app/config.py`
- Create: `backend/tests/conftest.py`
- Create: `backend/tests/test_config.py`

- [ ] **Step 1: Write the failing test**

`backend/tests/test_config.py`:
```python
import pytest
from app.config import Settings, get_settings


def test_default_environment():
    s = Settings()
    assert s.environment == "development"


def test_default_collection():
    s = Settings()
    assert s.qdrant_collection == "hr_documents"


def test_default_embedding_model():
    s = Settings()
    assert s.embedding_model == "paraphrase-multilingual-mpnet-base-v2"


def test_override_via_env(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "production")
    s = Settings()
    assert s.environment == "production"


def test_get_settings_returns_same_instance():
    get_settings.cache_clear()
    s1 = get_settings()
    s2 = get_settings()
    assert s1 is s2
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd backend
pip install pydantic-settings python-dotenv pytest
pytest tests/test_config.py -v
```

Expected: `ModuleNotFoundError: No module named 'app.config'`

- [ ] **Step 3: Create `backend/tests/conftest.py`**

```python
import pytest


@pytest.fixture(autouse=True)
def clear_settings_cache():
    from app.config import get_settings
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
```

- [ ] **Step 4: Create `backend/app/config.py`**

```python
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    environment: str = "development"

    # Ollama (dev)
    ollama_base_url: str = "http://host.docker.internal:11434"
    ollama_model: str = "qwen2.5:7b"

    # Claude (prod)
    anthropic_api_key: str = ""
    claude_model: str = "claude-sonnet-4-6"

    # Qdrant — backend container uses Docker service name
    qdrant_host: str = "qdrant"
    qdrant_port: int = 6333
    qdrant_collection: str = "hr_documents"

    # Qdrant — ingest script uses localhost (runs on host)
    qdrant_external_host: str = "localhost"

    # Embedding — identical in dev and prod
    embedding_model: str = "paraphrase-multilingual-mpnet-base-v2"


@lru_cache
def get_settings() -> Settings:
    return Settings()
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
cd backend
pytest tests/test_config.py -v
```

Expected: 5 passed

- [ ] **Step 6: Commit**

```bash
git add backend/app/config.py backend/tests/conftest.py backend/tests/test_config.py
git commit -m "feat: backend config with pydantic-settings"
```

---

## Task 3: Document Parser

**Files:**
- Create: `backend/app/parser.py`
- Create: `backend/tests/test_parser.py`

- [ ] **Step 1: Write the failing tests**

`backend/tests/test_parser.py`:
```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd backend
pytest tests/test_parser.py -v
```

Expected: `ModuleNotFoundError: No module named 'app.parser'`

- [ ] **Step 3: Create `backend/app/parser.py`**

```python
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
```

- [ ] **Step 4: Install dependencies and run tests**

```bash
cd backend
pip install pdfplumber python-docx
pytest tests/test_parser.py -v
```

Expected: 9 passed

- [ ] **Step 5: Commit**

```bash
git add backend/app/parser.py backend/tests/test_parser.py
git commit -m "feat: PDF and DOCX parser with table extraction"
```

---

## Task 4: Ingestion Script

**Files:**
- Create: `backend/app/ingest.py`
- Create: `backend/tests/test_ingest.py`

- [ ] **Step 1: Write the failing tests**

`backend/tests/test_ingest.py`:
```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd backend
pytest tests/test_ingest.py -v
```

Expected: `ModuleNotFoundError: No module named 'app.ingest'`

- [ ] **Step 3: Create `backend/app/ingest.py`**

```python
#!/usr/bin/env python3
"""
Indexes HR documents into Qdrant. Run from the project root:
    python backend/app/ingest.py

Wipes and rebuilds the collection on every run.
"""
import sys
from pathlib import Path

# Allow running as a script: adds backend/ to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import uuid

from app.config import get_settings
from app.parser import extract_from_pdf, extract_from_docx

DOCUMENTS_DIR = Path(__file__).parent.parent / "documents"

VECTOR_SIZE = 768  # paraphrase-multilingual-mpnet-base-v2 output dimension


def run_ingest() -> None:
    settings = get_settings()

    client = QdrantClient(host=settings.qdrant_external_host, port=settings.qdrant_port)
    model = SentenceTransformer(settings.embedding_model)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)

    # Wipe and recreate collection
    if client.collection_exists(settings.qdrant_collection):
        client.delete_collection(settings.qdrant_collection)
    client.create_collection(
        collection_name=settings.qdrant_collection,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )

    points = []
    for file_path in DOCUMENTS_DIR.iterdir():
        suffix = file_path.suffix.lower()
        if suffix == ".pdf":
            pages = list(extract_from_pdf(file_path))
        elif suffix == ".docx":
            pages = list(extract_from_docx(file_path))
        else:
            continue

        for page_data in pages:
            for chunk in splitter.split_text(page_data["text"]):
                vector = model.encode(chunk).tolist()
                points.append(PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vector,
                    payload={
                        "text": chunk,
                        "source": page_data["source"],
                        "page": page_data["page"],
                    },
                ))

    if points:
        client.upsert(collection_name=settings.qdrant_collection, points=points)
        print(f"Indexed {len(points)} chunks from {DOCUMENTS_DIR}")
    else:
        print("No documents found to index.")


if __name__ == "__main__":
    run_ingest()
```

- [ ] **Step 4: Install dependencies and run tests**

```bash
cd backend
pip install qdrant-client sentence-transformers langchain
pytest tests/test_ingest.py -v
```

Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add backend/app/ingest.py backend/tests/test_ingest.py
git commit -m "feat: document ingestion script with Qdrant wipe-and-rebuild"
```

---

## Task 5: RAG Chat Logic

**Files:**
- Create: `backend/app/chat.py`
- Create: `backend/tests/test_chat.py`

- [ ] **Step 1: Write the failing tests**

`backend/tests/test_chat.py`:
```python
from unittest.mock import MagicMock, patch
from langchain.schema import Document
from app.chat import answer_question, HR_REDIRECT, SIMILARITY_THRESHOLD
from app.config import Settings


def make_settings(**kwargs):
    defaults = dict(
        environment="development",
        ollama_base_url="http://localhost:11434",
        ollama_model="qwen2.5:7b",
        anthropic_api_key="",
        claude_model="claude-sonnet-4-6",
        qdrant_host="localhost",
        qdrant_port=6333,
        qdrant_collection="hr_documents",
        qdrant_external_host="localhost",
        embedding_model="paraphrase-multilingual-mpnet-base-v2",
    )
    defaults.update(kwargs)
    return Settings(**defaults)


def make_doc(content, source="handbook.pdf", page=1):
    return Document(page_content=content, metadata={"source": source, "page": page})


def test_returns_answer_when_context_found():
    settings = make_settings()
    mock_embeddings = MagicMock()
    mock_client = MagicMock()
    doc = make_doc("Employees receive 10 vacation days per year.")

    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="You receive 10 vacation days.")

    with patch("app.chat.retrieve_chunks", return_value=[(doc, 0.85)]), \
         patch("app.chat.build_llm", return_value=mock_llm):
        result = answer_question("How many vacation days?", "en", [], settings, mock_embeddings, mock_client)

    assert result["answer"] == "You receive 10 vacation days."
    assert result["citations"] == [{"source": "handbook.pdf", "page": 1}]


def test_returns_hr_redirect_when_below_threshold():
    settings = make_settings()
    mock_embeddings = MagicMock()
    mock_client = MagicMock()
    doc = make_doc("Irrelevant content.")

    with patch("app.chat.retrieve_chunks", return_value=[(doc, SIMILARITY_THRESHOLD - 0.01)]):
        result = answer_question("What is the weather today?", "en", [], settings, mock_embeddings, mock_client)

    assert result["answer"] == HR_REDIRECT
    assert result["citations"] == []


def test_returns_hr_redirect_when_no_chunks():
    settings = make_settings()
    mock_embeddings = MagicMock()
    mock_client = MagicMock()

    with patch("app.chat.retrieve_chunks", return_value=[]):
        result = answer_question("Random question", "en", [], settings, mock_embeddings, mock_client)

    assert result["answer"] == HR_REDIRECT
    assert result["citations"] == []


def test_deduplicates_citations():
    settings = make_settings()
    mock_embeddings = MagicMock()
    mock_client = MagicMock()
    doc1 = make_doc("Benefits info.", source="benefits.pdf", page=3)
    doc2 = make_doc("More benefits.", source="benefits.pdf", page=3)

    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="Here are your benefits.")

    with patch("app.chat.retrieve_chunks", return_value=[(doc1, 0.9), (doc2, 0.8)]), \
         patch("app.chat.build_llm", return_value=mock_llm):
        result = answer_question("What are my benefits?", "en", [], settings, mock_embeddings, mock_client)

    assert len(result["citations"]) == 1
    assert result["citations"][0] == {"source": "benefits.pdf", "page": 3}


def test_includes_conversation_history_in_llm_call():
    settings = make_settings()
    mock_embeddings = MagicMock()
    mock_client = MagicMock()
    doc = make_doc("FMLA allows 12 weeks of leave.")

    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="FMLA gives you 12 weeks.")

    history = [
        {"role": "user", "content": "What is FMLA?"},
        {"role": "assistant", "content": "FMLA is the Family and Medical Leave Act."},
    ]

    with patch("app.chat.retrieve_chunks", return_value=[(doc, 0.9)]), \
         patch("app.chat.build_llm", return_value=mock_llm):
        answer_question("How many weeks does it cover?", "en", history, settings, mock_embeddings, mock_client)

    call_args = mock_llm.invoke.call_args[0][0]
    # Should have system + 2 history messages + new user message = 4
    assert len(call_args) == 4


def test_uses_spanish_in_system_prompt():
    settings = make_settings()
    mock_embeddings = MagicMock()
    mock_client = MagicMock()
    doc = make_doc("Employees receive 10 vacation days.")

    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="Tienes 10 días de vacaciones.")

    with patch("app.chat.retrieve_chunks", return_value=[(doc, 0.9)]), \
         patch("app.chat.build_llm", return_value=mock_llm):
        answer_question("¿Cuántos días de vacaciones?", "es", [], settings, mock_embeddings, mock_client)

    system_msg = mock_llm.invoke.call_args[0][0][0]
    assert "Spanish" in system_msg.content
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd backend
pytest tests/test_chat.py -v
```

Expected: `ModuleNotFoundError: No module named 'app.chat'`

- [ ] **Step 3: Install LangChain dependencies**

```bash
pip install langchain-community langchain-anthropic langchain-ollama langchain-huggingface
```

- [ ] **Step 4: Create `backend/app/chat.py`**

```python
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain_community.vectorstores import Qdrant as QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient

from app.config import Settings

SIMILARITY_THRESHOLD = 0.3

HR_REDIRECT = (
    "I'm sorry, I don't have information about that in the HR documents. "
    "Please contact HR directly: [HR_NAME], [HR_EMAIL], [HR_PHONE]."
)

LANGUAGE_NAMES = {
    "en": "English",
    "es": "Spanish",
    "ps": "Pashto",
    "fa": "Dari",
    "bs": "Bosnian",
}

SYSTEM_TEMPLATE = (
    "You are an HR assistant for General Stamping & Metalworks (GSM).\n"
    "Answer the employee's question using ONLY the context provided below.\n"
    "Do not use any knowledge outside of this context.\n"
    "If the context does not contain enough information to answer the question, "
    "say you don't know and direct the employee to HR.\n"
    "Respond in {language}.\n\n"
    "Context:\n{context}"
)


def build_llm(settings: Settings):
    if settings.environment == "development":
        from langchain_ollama import ChatOllama
        return ChatOllama(base_url=settings.ollama_base_url, model=settings.ollama_model)
    from langchain_anthropic import ChatAnthropic
    return ChatAnthropic(
        model=settings.claude_model,
        anthropic_api_key=settings.anthropic_api_key,
    )


def build_embeddings(settings: Settings) -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=settings.embedding_model)


def get_qdrant_client(settings: Settings) -> QdrantClient:
    return QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)


def retrieve_chunks(
    query: str,
    settings: Settings,
    embeddings: HuggingFaceEmbeddings,
    client: QdrantClient,
) -> list[tuple]:
    """Returns list of (Document, score) tuples, sorted by descending score."""
    store = QdrantVectorStore(
        client=client,
        collection_name=settings.qdrant_collection,
        embeddings=embeddings,
    )
    return store.similarity_search_with_score(query, k=5)


def answer_question(
    message: str,
    language: str,
    history: list[dict],
    settings: Settings,
    embeddings: HuggingFaceEmbeddings,
    client: QdrantClient,
) -> dict:
    results = retrieve_chunks(message, settings, embeddings, client)
    relevant = [(doc, score) for doc, score in results if score >= SIMILARITY_THRESHOLD]

    if not relevant:
        return {"answer": HR_REDIRECT, "citations": []}

    context = "\n\n".join(doc.page_content for doc, _ in relevant)
    lang_name = LANGUAGE_NAMES.get(language, "English")
    system_prompt = SYSTEM_TEMPLATE.format(context=context, language=lang_name)

    # Build deduplicated citations from chunk metadata
    seen: set[tuple] = set()
    citations = []
    for doc, _ in relevant:
        key = (doc.metadata.get("source", ""), doc.metadata.get("page", 0))
        if key not in seen:
            seen.add(key)
            citations.append({"source": key[0], "page": key[1]})

    # Build message list for LLM
    messages = [SystemMessage(content=system_prompt)]
    for turn in history:
        if turn["role"] == "user":
            messages.append(HumanMessage(content=turn["content"]))
        else:
            messages.append(AIMessage(content=turn["content"]))
    messages.append(HumanMessage(content=message))

    llm = build_llm(settings)
    response = llm.invoke(messages)
    answer = response.content if hasattr(response, "content") else str(response)

    return {"answer": answer, "citations": citations}
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
cd backend
pytest tests/test_chat.py -v
```

Expected: 6 passed

- [ ] **Step 6: Commit**

```bash
git add backend/app/chat.py backend/tests/test_chat.py
git commit -m "feat: RAG chat logic with LLM switching and citation extraction"
```

---

## Task 6: FastAPI App

**Files:**
- Create: `backend/app/main.py`
- Create: `backend/tests/test_main.py`

- [ ] **Step 1: Write the failing tests**

`backend/tests/test_main.py`:
```python
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient


def test_health():
    from app.main import app
    client = TestClient(app)
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_chat_returns_answer_and_citations():
    mock_result = {
        "answer": "You receive 10 vacation days.",
        "citations": [{"source": "handbook.pdf", "page": 4}],
    }
    with patch("app.main.answer_question", return_value=mock_result), \
         patch("app.main.get_embeddings", return_value=MagicMock()), \
         patch("app.main.get_client", return_value=MagicMock()):
        from app.main import app
        client = TestClient(app)
        response = client.post("/api/chat", json={
            "message": "How many vacation days?",
            "language": "en",
            "history": [],
        })

    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "You receive 10 vacation days."
    assert data["citations"][0]["source"] == "handbook.pdf"
    assert data["citations"][0]["page"] == 4


def test_chat_defaults_language_to_en():
    mock_result = {"answer": "Contact HR.", "citations": []}
    with patch("app.main.answer_question", return_value=mock_result) as mock_fn, \
         patch("app.main.get_embeddings", return_value=MagicMock()), \
         patch("app.main.get_client", return_value=MagicMock()):
        from app.main import app
        client = TestClient(app)
        client.post("/api/chat", json={"message": "What is FMLA?"})

    call_kwargs = mock_fn.call_args.kwargs
    assert call_kwargs["language"] == "en"


def test_chat_passes_history():
    mock_result = {"answer": "It covers 12 weeks.", "citations": []}
    history = [
        {"role": "user", "content": "What is FMLA?"},
        {"role": "assistant", "content": "It is the Family Medical Leave Act."},
    ]
    with patch("app.main.answer_question", return_value=mock_result) as mock_fn, \
         patch("app.main.get_embeddings", return_value=MagicMock()), \
         patch("app.main.get_client", return_value=MagicMock()):
        from app.main import app
        client = TestClient(app)
        client.post("/api/chat", json={
            "message": "How long does it last?",
            "language": "en",
            "history": history,
        })

    call_kwargs = mock_fn.call_args.kwargs
    assert len(call_kwargs["history"]) == 2
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd backend
pytest tests/test_main.py -v
```

Expected: `ModuleNotFoundError: No module named 'app.main'`

- [ ] **Step 3: Create `backend/app/main.py`**

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.config import get_settings
from app.chat import answer_question, build_embeddings, get_qdrant_client

app = FastAPI(title="GSM HR Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

_settings = get_settings()
_embeddings = None
_client = None


def get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = build_embeddings(_settings)
    return _embeddings


def get_client():
    global _client
    if _client is None:
        _client = get_qdrant_client(_settings)
    return _client


class HistoryItem(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str
    language: str = "en"
    history: list[HistoryItem] = []


class Citation(BaseModel):
    source: str
    page: int


class ChatResponse(BaseModel):
    answer: str
    citations: list[Citation]


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    result = answer_question(
        message=req.message,
        language=req.language,
        history=[item.model_dump() for item in req.history],
        settings=_settings,
        embeddings=get_embeddings(),
        client=get_client(),
    )
    return ChatResponse(**result)
```

- [ ] **Step 4: Install httpx and run tests**

```bash
cd backend
pip install fastapi uvicorn httpx
pytest tests/test_main.py -v
```

Expected: 4 passed

- [ ] **Step 5: Run all backend tests**

```bash
cd backend
pytest tests/ -v
```

Expected: all tests pass

- [ ] **Step 6: Commit**

```bash
git add backend/app/main.py backend/tests/test_main.py
git commit -m "feat: FastAPI app with /api/chat and /api/health endpoints"
```

---

## Task 7: Backend requirements.txt and Dockerfile

**Files:**
- Create: `backend/requirements.txt`
- Create: `backend/Dockerfile`

- [ ] **Step 1: Create `backend/requirements.txt`**

```
fastapi>=0.115.0
uvicorn[standard]>=0.30.0
pydantic-settings>=2.5.0
langchain>=0.3.0
langchain-community>=0.3.0
langchain-anthropic>=0.3.0
langchain-ollama>=0.2.0
langchain-huggingface>=0.1.0
sentence-transformers>=3.0.0
qdrant-client>=1.9.0
pdfplumber>=0.11.0
python-docx>=1.1.0
python-dotenv>=1.0.0
pytest>=8.0.0
httpx>=0.27.0
```

- [ ] **Step 2: Create `backend/Dockerfile`**

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

- [ ] **Step 3: Verify Docker build succeeds**

```bash
docker build -t hr-chatbot-backend ./backend
```

Expected: Successfully built (no errors)

- [ ] **Step 4: Commit**

```bash
git add backend/requirements.txt backend/Dockerfile
git commit -m "chore: backend requirements and Dockerfile"
```

---

## Task 8: Frontend Scaffold

**Files:**
- Create: `frontend/package.json`
- Create: `frontend/vite.config.js`
- Create: `frontend/public/index.html`
- Create: `frontend/src/main.jsx`
- Create: `frontend/src/setupTests.js`

- [ ] **Step 1: Create `frontend/package.json`**

```json
{
  "name": "hr-chatbot-frontend",
  "version": "1.0.0",
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "vite build",
    "preview": "vite preview",
    "test": "vitest run"
  },
  "dependencies": {
    "react": "^18.3.1",
    "react-dom": "^18.3.1"
  },
  "devDependencies": {
    "@vitejs/plugin-react": "^4.3.1",
    "@testing-library/react": "^16.0.1",
    "@testing-library/jest-dom": "^6.4.6",
    "jsdom": "^24.1.1",
    "vite": "^5.4.2",
    "vitest": "^2.0.5"
  }
}
```

- [ ] **Step 2: Create `frontend/vite.config.js`**

```js
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/api': 'http://localhost:8000',
    },
  },
  test: {
    globals: true,
    environment: 'jsdom',
    setupFiles: './src/setupTests.js',
  },
})
```

- [ ] **Step 3: Create `frontend/src/setupTests.js`**

```js
import '@testing-library/jest-dom'
```

- [ ] **Step 4: Create `frontend/public/index.html`**

```html
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>GSM HR Assistant</title>
  </head>
  <body>
    <div id="root"></div>
    <script type="module" src="/src/main.jsx"></script>
  </body>
</html>
```

- [ ] **Step 5: Create `frontend/src/main.jsx`**

```jsx
import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App'

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
)
```

- [ ] **Step 6: Install dependencies**

```bash
cd frontend
npm install
```

Expected: `node_modules/` created, no errors

- [ ] **Step 7: Commit**

```bash
git add frontend/package.json frontend/vite.config.js frontend/public/index.html frontend/src/main.jsx frontend/src/setupTests.js
git commit -m "chore: frontend scaffold with Vite and React"
```

---

## Task 9: Frontend API Client

**Files:**
- Create: `frontend/src/api.js`
- Create: `frontend/src/__tests__/api.test.js`

- [ ] **Step 1: Write the failing test**

`frontend/src/__tests__/api.test.js`:
```js
import { sendMessage } from '../api'

afterEach(() => {
  vi.restoreAllMocks()
})

test('sendMessage posts to /api/chat with correct body', async () => {
  global.fetch = vi.fn().mockResolvedValue({
    ok: true,
    json: async () => ({ answer: 'Ten days.', citations: [] }),
  })

  const result = await sendMessage({ message: 'Vacation?', language: 'en', history: [] })

  expect(global.fetch).toHaveBeenCalledWith('/api/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message: 'Vacation?', language: 'en', history: [] }),
  })
  expect(result.answer).toBe('Ten days.')
  expect(result.citations).toEqual([])
})

test('sendMessage throws on non-ok response', async () => {
  global.fetch = vi.fn().mockResolvedValue({ ok: false, status: 500 })

  await expect(
    sendMessage({ message: 'Vacation?', language: 'en', history: [] })
  ).rejects.toThrow('Server error: 500')
})
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd frontend
npm test
```

Expected: `Cannot find module '../api'`

- [ ] **Step 3: Create `frontend/src/api.js`**

```js
export async function sendMessage({ message, language, history }) {
  const response = await fetch('/api/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message, language, history }),
  })
  if (!response.ok) {
    throw new Error(`Server error: ${response.status}`)
  }
  return response.json()
}
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd frontend
npm test
```

Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add frontend/src/api.js frontend/src/__tests__/api.test.js
git commit -m "feat: frontend API client"
```

---

## Task 10: LanguageDropdown Component

**Files:**
- Create: `frontend/src/components/LanguageDropdown.jsx`
- Create: `frontend/src/__tests__/LanguageDropdown.test.jsx`

- [ ] **Step 1: Write the failing test**

`frontend/src/__tests__/LanguageDropdown.test.jsx`:
```jsx
import { render, screen, fireEvent } from '@testing-library/react'
import LanguageDropdown from '../components/LanguageDropdown'

test('renders all five language options', () => {
  render(<LanguageDropdown value="en" onChange={() => {}} />)
  expect(screen.getByText('English')).toBeInTheDocument()
  expect(screen.getByText('Español')).toBeInTheDocument()
  expect(screen.getByText('پښتو')).toBeInTheDocument()
  expect(screen.getByText('دری')).toBeInTheDocument()
  expect(screen.getByText('Bosanski')).toBeInTheDocument()
})

test('calls onChange with selected language code', () => {
  const onChange = vi.fn()
  render(<LanguageDropdown value="en" onChange={onChange} />)
  fireEvent.change(screen.getByRole('combobox'), { target: { value: 'es' } })
  expect(onChange).toHaveBeenCalledWith('es')
})

test('shows current value as selected', () => {
  render(<LanguageDropdown value="bs" onChange={() => {}} />)
  expect(screen.getByRole('combobox')).toHaveValue('bs')
})
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd frontend
npm test
```

Expected: `Cannot find module '../components/LanguageDropdown'`

- [ ] **Step 3: Create `frontend/src/components/LanguageDropdown.jsx`**

```jsx
const LANGUAGES = [
  { code: 'en', label: 'English' },
  { code: 'es', label: 'Español' },
  { code: 'ps', label: 'پښتو' },
  { code: 'fa', label: 'دری' },
  { code: 'bs', label: 'Bosanski' },
]

export default function LanguageDropdown({ value, onChange }) {
  return (
    <select value={value} onChange={e => onChange(e.target.value)}>
      {LANGUAGES.map(lang => (
        <option key={lang.code} value={lang.code}>
          {lang.label}
        </option>
      ))}
    </select>
  )
}
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd frontend
npm test
```

Expected: all tests pass

- [ ] **Step 5: Commit**

```bash
git add frontend/src/components/LanguageDropdown.jsx frontend/src/__tests__/LanguageDropdown.test.jsx
git commit -m "feat: LanguageDropdown component"
```

---

## Task 11: MessageBubble Component

**Files:**
- Create: `frontend/src/components/MessageBubble.jsx`
- Create: `frontend/src/__tests__/MessageBubble.test.jsx`

- [ ] **Step 1: Write the failing test**

`frontend/src/__tests__/MessageBubble.test.jsx`:
```jsx
import { render, screen } from '@testing-library/react'
import MessageBubble from '../components/MessageBubble'

test('renders message content', () => {
  render(<MessageBubble role="user" content="Hello there" citations={[]} />)
  expect(screen.getByText('Hello there')).toBeInTheDocument()
})

test('applies user class for user messages', () => {
  const { container } = render(
    <MessageBubble role="user" content="Hi" citations={[]} />
  )
  expect(container.firstChild).toHaveClass('message', 'user')
})

test('applies assistant class for assistant messages', () => {
  const { container } = render(
    <MessageBubble role="assistant" content="Hi" citations={[]} />
  )
  expect(container.firstChild).toHaveClass('message', 'assistant')
})

test('renders citations when present', () => {
  const citations = [
    { source: 'handbook.pdf', page: 5 },
    { source: 'benefits.pdf', page: 2 },
  ]
  render(<MessageBubble role="assistant" content="Here is the info." citations={citations} />)
  expect(screen.getByText(/handbook.pdf/)).toBeInTheDocument()
  expect(screen.getByText(/p\. 5/)).toBeInTheDocument()
  expect(screen.getByText(/benefits.pdf/)).toBeInTheDocument()
})

test('does not render citations section when empty', () => {
  render(<MessageBubble role="assistant" content="Contact HR." citations={[]} />)
  expect(screen.queryByText(/Sources:/)).not.toBeInTheDocument()
})
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd frontend
npm test
```

Expected: `Cannot find module '../components/MessageBubble'`

- [ ] **Step 3: Create `frontend/src/components/MessageBubble.jsx`**

```jsx
export default function MessageBubble({ role, content, citations }) {
  return (
    <div className={`message ${role}`}>
      <p>{content}</p>
      {citations && citations.length > 0 && (
        <p className="citations">
          {'Sources: '}
          {citations.map((c, i) => (
            <span key={i}>
              {c.source} (p. {c.page}){i < citations.length - 1 ? ', ' : ''}
            </span>
          ))}
        </p>
      )}
    </div>
  )
}
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd frontend
npm test
```

Expected: all tests pass

- [ ] **Step 5: Commit**

```bash
git add frontend/src/components/MessageBubble.jsx frontend/src/__tests__/MessageBubble.test.jsx
git commit -m "feat: MessageBubble component with citation display"
```

---

## Task 12: InputBar Component

**Files:**
- Create: `frontend/src/components/InputBar.jsx`
- Create: `frontend/src/__tests__/InputBar.test.jsx`

- [ ] **Step 1: Write the failing test**

`frontend/src/__tests__/InputBar.test.jsx`:
```jsx
import { render, screen, fireEvent } from '@testing-library/react'
import InputBar from '../components/InputBar'

test('renders input and send button', () => {
  render(<InputBar onSend={() => {}} disabled={false} />)
  expect(screen.getByRole('textbox')).toBeInTheDocument()
  expect(screen.getByRole('button', { name: /send/i })).toBeInTheDocument()
})

test('calls onSend with trimmed message on submit', () => {
  const onSend = vi.fn()
  render(<InputBar onSend={onSend} disabled={false} />)
  fireEvent.change(screen.getByRole('textbox'), { target: { value: '  Hello  ' } })
  fireEvent.click(screen.getByRole('button', { name: /send/i }))
  expect(onSend).toHaveBeenCalledWith('Hello')
})

test('clears input after submit', () => {
  render(<InputBar onSend={() => {}} disabled={false} />)
  const input = screen.getByRole('textbox')
  fireEvent.change(input, { target: { value: 'Hello' } })
  fireEvent.click(screen.getByRole('button', { name: /send/i }))
  expect(input).toHaveValue('')
})

test('does not call onSend for empty or whitespace input', () => {
  const onSend = vi.fn()
  render(<InputBar onSend={onSend} disabled={false} />)
  fireEvent.change(screen.getByRole('textbox'), { target: { value: '   ' } })
  fireEvent.click(screen.getByRole('button', { name: /send/i }))
  expect(onSend).not.toHaveBeenCalled()
})

test('disables input and button when disabled prop is true', () => {
  render(<InputBar onSend={() => {}} disabled={true} />)
  expect(screen.getByRole('textbox')).toBeDisabled()
  expect(screen.getByRole('button', { name: /send/i })).toBeDisabled()
})
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd frontend
npm test
```

Expected: `Cannot find module '../components/InputBar'`

- [ ] **Step 3: Create `frontend/src/components/InputBar.jsx`**

```jsx
import { useState } from 'react'

export default function InputBar({ onSend, disabled }) {
  const [input, setInput] = useState('')

  function handleSubmit(e) {
    e.preventDefault()
    if (!input.trim()) return
    onSend(input.trim())
    setInput('')
  }

  return (
    <form onSubmit={handleSubmit} className="input-bar">
      <input
        type="text"
        value={input}
        onChange={e => setInput(e.target.value)}
        disabled={disabled}
        placeholder="Type your question..."
      />
      <button type="submit" disabled={disabled || !input.trim()}>
        Send
      </button>
    </form>
  )
}
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd frontend
npm test
```

Expected: all tests pass

- [ ] **Step 5: Commit**

```bash
git add frontend/src/components/InputBar.jsx frontend/src/__tests__/InputBar.test.jsx
git commit -m "feat: InputBar component"
```

---

## Task 13: ChatWindow Component

**Files:**
- Create: `frontend/src/components/ChatWindow.jsx`

No dedicated test — ChatWindow delegates rendering to MessageBubble (already tested) and manages scroll-to-bottom via a ref, which is a DOM side effect not worth unit testing.

- [ ] **Step 1: Create `frontend/src/components/ChatWindow.jsx`**

```jsx
import { useEffect, useRef } from 'react'
import MessageBubble from './MessageBubble'

export default function ChatWindow({ messages }) {
  const bottomRef = useRef(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  return (
    <div className="chat-window">
      {messages.map((msg, i) => (
        <MessageBubble
          key={i}
          role={msg.role}
          content={msg.content}
          citations={msg.citations}
        />
      ))}
      <div ref={bottomRef} />
    </div>
  )
}
```

- [ ] **Step 2: Commit**

```bash
git add frontend/src/components/ChatWindow.jsx
git commit -m "feat: ChatWindow component"
```

---

## Task 14: App.jsx and App.css

**Files:**
- Create: `frontend/src/App.jsx`
- Create: `frontend/src/App.css`

- [ ] **Step 1: Create `frontend/src/App.css`**

```css
* {
  box-sizing: border-box;
  margin: 0;
  padding: 0;
}

body {
  font-family: sans-serif;
  background: #f5f5f5;
}

.app {
  display: flex;
  flex-direction: column;
  height: 100vh;
  max-width: 800px;
  margin: 0 auto;
  background: white;
  box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
}

header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 1rem;
  border-bottom: 1px solid #ddd;
  background: white;
}

header h1 {
  font-size: 1.2rem;
  color: #333;
}

header select {
  padding: 0.3rem 0.5rem;
  font-size: 0.9rem;
  border: 1px solid #ddd;
  border-radius: 4px;
}

.chat-window {
  flex: 1;
  overflow-y: auto;
  padding: 1rem;
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
}

.message {
  max-width: 75%;
  padding: 0.75rem 1rem;
  border-radius: 12px;
  line-height: 1.5;
}

.message.user {
  align-self: flex-end;
  background: #007bff;
  color: white;
}

.message.assistant {
  align-self: flex-start;
  background: #e9ecef;
  color: #333;
}

.citations {
  margin-top: 0.5rem;
  font-size: 0.78rem;
  opacity: 0.7;
}

.input-bar {
  display: flex;
  padding: 1rem;
  border-top: 1px solid #ddd;
  gap: 0.5rem;
  background: white;
}

.input-bar input {
  flex: 1;
  padding: 0.5rem 0.75rem;
  border: 1px solid #ddd;
  border-radius: 6px;
  font-size: 1rem;
}

.input-bar button {
  padding: 0.5rem 1.25rem;
  background: #007bff;
  color: white;
  border: none;
  border-radius: 6px;
  cursor: pointer;
  font-size: 1rem;
}

.input-bar button:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}
```

- [ ] **Step 2: Create `frontend/src/App.jsx`**

```jsx
import { useState } from 'react'
import LanguageDropdown from './components/LanguageDropdown'
import ChatWindow from './components/ChatWindow'
import InputBar from './components/InputBar'
import { sendMessage } from './api'
import './App.css'

export default function App() {
  const [language, setLanguage] = useState('en')
  const [messages, setMessages] = useState([])
  const [loading, setLoading] = useState(false)

  async function handleSend(text) {
    const newUserMsg = { role: 'user', content: text, citations: [] }
    const updatedMessages = [...messages, newUserMsg]
    setMessages(updatedMessages)
    setLoading(true)

    try {
      const history = updatedMessages.slice(0, -1).map(m => ({
        role: m.role === 'user' ? 'user' : 'assistant',
        content: m.content,
      }))
      const result = await sendMessage({ message: text, language, history })
      setMessages(prev => [
        ...prev,
        { role: 'assistant', content: result.answer, citations: result.citations },
      ])
    } catch {
      setMessages(prev => [
        ...prev,
        {
          role: 'assistant',
          content: 'Something went wrong. Please try again.',
          citations: [],
        },
      ])
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="app">
      <header>
        <h1>GSM HR Assistant</h1>
        <LanguageDropdown value={language} onChange={setLanguage} />
      </header>
      <ChatWindow messages={messages} />
      <InputBar onSend={handleSend} disabled={loading} />
    </div>
  )
}
```

- [ ] **Step 3: Run all frontend tests**

```bash
cd frontend
npm test
```

Expected: all tests pass

- [ ] **Step 4: Verify dev server starts**

```bash
cd frontend
npm run dev
```

Expected: Vite server running at `http://localhost:5173` (no errors in console)

Stop the server with Ctrl+C.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/App.jsx frontend/src/App.css
git commit -m "feat: App component wiring chat UI together"
```

---

## Task 15: Frontend Dockerfile and Nginx Config

**Files:**
- Create: `frontend/nginx.conf`
- Create: `frontend/Dockerfile`

- [ ] **Step 1: Create `frontend/nginx.conf`**

```nginx
server {
    listen 3000;
    root /usr/share/nginx/html;
    index index.html;

    location /api/ {
        proxy_pass http://backend:8000/api/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }

    location / {
        try_files $uri $uri/ /index.html;
    }
}
```

- [ ] **Step 2: Create `frontend/Dockerfile`**

```dockerfile
FROM node:20-alpine AS builder
WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .
RUN npm run build

FROM nginx:alpine
COPY --from=builder /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 3000
```

- [ ] **Step 3: Verify Docker build succeeds**

```bash
docker build -t hr-chatbot-frontend ./frontend
```

Expected: Successfully built (no errors)

- [ ] **Step 4: Commit**

```bash
git add frontend/nginx.conf frontend/Dockerfile
git commit -m "chore: frontend Dockerfile and nginx config"
```

---

## Task 16: Docker Compose

**Files:**
- Create: `docker-compose.yml`

- [ ] **Step 1: Create `docker-compose.yml`**

```yaml
version: "3.9"

services:
  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
    volumes:
      - qdrant_data:/qdrant/storage

  backend:
    build: ./backend
    ports:
      - "8000:8000"
    env_file: .env
    depends_on:
      - qdrant
    extra_hosts:
      - "host.docker.internal:host-gateway"
    volumes:
      - ./backend/documents:/app/documents

  frontend:
    build: ./frontend
    ports:
      - "3000:3000"
    depends_on:
      - backend

volumes:
  qdrant_data:
```

- [ ] **Step 2: Commit**

```bash
git add docker-compose.yml
git commit -m "chore: Docker Compose config"
```

---

## Task 17: End-to-End Smoke Test

Verify the full stack works together: start Docker Compose, run ingestion, and confirm the chat endpoint returns a grounded answer.

- [ ] **Step 1: Drop an HR document into `backend/documents/`**

Copy at least one PDF (e.g., `GSM Handbook 2026.pdf`) into `backend/documents/`.

- [ ] **Step 2: Start Docker Compose**

```bash
docker compose up --build -d
```

Wait ~30 seconds for all services to start. Verify all three containers are running:

```bash
docker compose ps
```

Expected: `qdrant`, `backend`, and `frontend` all show `running`.

- [ ] **Step 3: Confirm backend health**

```bash
curl http://localhost:8000/api/health
```

Expected: `{"status":"ok"}`

- [ ] **Step 4: Run the ingestion script**

```bash
python backend/app/ingest.py
```

Expected output: `Indexed N chunks from .../backend/documents`

Note: The embedding model (~420 MB) downloads on first run. This is expected.

- [ ] **Step 5: Test the chat endpoint**

```bash
curl -s -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is the vacation policy?", "language": "en", "history": []}' \
  | python3 -m json.tool
```

Expected: JSON response with `answer` (grounded in document text) and `citations` (with source filename and page number). If no relevant chunks found, `answer` will contain the HR redirect message — this is correct behavior.

- [ ] **Step 6: Open the UI**

Open `http://localhost:3000` in a browser.

Verify:
- Language dropdown shows 5 options
- Typing a question and pressing Send shows a user bubble on the right
- A response appears on the left with citations beneath it

- [ ] **Step 7: Test HR redirect**

Ask a question with no HR relevance, e.g. "What is the capital of France?"

Expected: Response contains the HR redirect placeholder message, no citations.

- [ ] **Step 8: Stop the stack**

```bash
docker compose down
```

- [ ] **Step 9: Commit**

```bash
git add .
git commit -m "chore: end-to-end smoke test verified"
```

---

## Running Tests

**Backend (from `backend/` directory):**
```bash
pytest tests/ -v
```

**Frontend (from `frontend/` directory):**
```bash
npm test
```
