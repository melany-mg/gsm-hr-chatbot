#!/usr/bin/env python3
"""
Diagnostic script for RAG retrieval failures.

Checks three things:
  1. What pdfplumber extracts from page 41 of GSM Handbook 2026.pdf
     (exposes the double-extraction bug)
  2. What chunks are stored in Qdrant for that page
  3. What similarity scores page-41 chunks get for the two failing queries

Run inside the container:
    docker compose exec backend python debug_retrieval.py

Or locally (with Qdrant on localhost):
    QDRANT_HOST=localhost python backend/debug_retrieval.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import pdfplumber
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

from app.config import get_settings

HANDBOOK = Path(__file__).parent / "documents" / "GSM Handbook 2026.pdf"
TARGET_PAGE = 41

FAILING_QUERIES = [
    "Can you show me how much vacation we earn based on employment length?",
    "What if I've worked here for two years?",
]

SEP = "─" * 72


def section(title: str) -> None:
    print(f"\n{SEP}\n  {title}\n{SEP}")


# ── 1. PDF extraction ────────────────────────────────────────────────────────

section(f"PDF extraction — page {TARGET_PAGE}")

with pdfplumber.open(HANDBOOK) as pdf:
    page = pdf.pages[TARGET_PAGE - 1]  # 0-indexed

    raw_text = page.extract_text() or ""
    raw_tables = page.extract_tables()

print(f"\n[extract_text() — {len(raw_text)} chars]\n")
print(raw_text[:2000])

print(f"\n[extract_tables() — {len(raw_tables)} table(s)]\n")
for t_idx, table in enumerate(raw_tables):
    print(f"  Table {t_idx + 1}: {len(table)} rows × {len(table[0]) if table else 0} cols")
    for row in table:
        print("  ", " | ".join(str(c) if c is not None else "" for c in row))

# ── 2. Qdrant chunks for page 41 ─────────────────────────────────────────────

section(f"Qdrant chunks — page {TARGET_PAGE}")

settings = get_settings()
client = QdrantClient(host=settings.qdrant_external_host, port=settings.qdrant_port)

scroll_filter = Filter(
    must=[FieldCondition(key="metadata.page", match=MatchValue(value=TARGET_PAGE))]
)
results, _ = client.scroll(
    collection_name=settings.qdrant_collection,
    scroll_filter=scroll_filter,
    limit=50,
    with_payload=True,
    with_vectors=False,
)

if not results:
    print(f"\n  ⚠  No chunks found for page {TARGET_PAGE} in collection "
          f"'{settings.qdrant_collection}'.")
else:
    print(f"\n  Found {len(results)} chunk(s) for page {TARGET_PAGE}:\n")
    for i, pt in enumerate(results, 1):
        content = pt.payload.get("page_content", "")
        print(f"  [{i}] {len(content)} chars — {content[:120]!r}…\n")

# ── 3. Similarity search scores ──────────────────────────────────────────────

from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name=settings.embedding_model)
store = QdrantVectorStore(
    client=client,
    collection_name=settings.qdrant_collection,
    embedding=embeddings,
)

for query in FAILING_QUERIES:
    section(f"Query: {query!r}")
    hits = store.similarity_search_with_score(query, k=20)
    print(f"\n  {'Score':>6}  {'Page':>5}  Preview")
    print(f"  {'─'*6}  {'─'*5}  {'─'*50}")
    for doc, score in hits:
        pg = doc.metadata.get("page", "?")
        marker = "  <<<" if pg == TARGET_PAGE else ""
        preview = doc.page_content[:60].replace("\n", " ")
        print(f"  {score:6.4f}  {str(pg):>5}  {preview!r}{marker}")

print(f"\n{SEP}\n  Done.\n{SEP}\n")
