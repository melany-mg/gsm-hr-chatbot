import csv
import logging
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.config import get_settings
from app.chat import answer_question, build_embeddings, get_qdrant_client

LOGS_DIR = Path("/app/logs")
LOGS_DIR.mkdir(parents=True, exist_ok=True)

LOG_PATH = LOGS_DIR / "usage.log"
CSV_PATH = LOGS_DIR / "usage.csv"

logging.basicConfig(
    filename=LOG_PATH,
    level=logging.INFO,
    format="%(message)s",
)
usage_logger = logging.getLogger("usage")

CSV_COLUMNS = ["Timestamp", "Session ID", "Language", "Question", "Answer", "Outcome", "Citations"]


def _migrate_csv() -> None:
    if not CSV_PATH.exists():
        with open(CSV_PATH, "w", newline="") as f:
            csv.writer(f).writerow(CSV_COLUMNS)
        return
    with open(CSV_PATH, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        existing_cols = list(reader.fieldnames or [])
        rows = list(reader)
    if existing_cols == CSV_COLUMNS:
        return
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in CSV_COLUMNS})


_migrate_csv()

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
    session_id: str = ""


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
    outcome = "redirected" if not result["citations"] else "answered"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    usage_logger.info(
        "%s | language=%s | outcome=%s | citations=%d",
        timestamp,
        req.language,
        outcome,
        len(result["citations"]),
    )
    write_header = not CSV_PATH.exists()
    with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(CSV_COLUMNS)
        writer.writerow([timestamp, req.session_id, req.language, req.message, result["answer"], outcome, len(result["citations"])])
    return ChatResponse(**result)
