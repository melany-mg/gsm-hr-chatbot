#!/usr/bin/env python3
"""
Indexes HR documents into Qdrant. Run from the project root:
    python backend/app/ingest.py

Wipes and rebuilds the collection on every run.

docker compose exec -e QDRANT_EXTERNAL_HOST=qdrant backend python app/ingest.py 2>&1 | tail -5
"""
import sys
from pathlib import Path

# Allow running as a script: adds backend/ to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
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
                raw = model.encode(chunk)
                vector = raw.tolist() if hasattr(raw, "tolist") else list(raw)
                points.append(PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vector,
                    payload={
                        "page_content": chunk,
                        "metadata": {
                            "source": page_data["source"],
                            "page": page_data["page"],
                        },
                    },
                ))

    if points:
        client.upsert(collection_name=settings.qdrant_collection, points=points)
        print(f"Indexed {len(points)} chunks from {DOCUMENTS_DIR}")
    else:
        print("No documents found to index.")


if __name__ == "__main__":
    run_ingest()
