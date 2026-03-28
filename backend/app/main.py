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
