import csv
import secrets
import threading
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, UploadFile
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

from app.config import get_settings

router = APIRouter(prefix="/api/admin", tags=["admin"])

_token: str = secrets.token_hex(24)  # 48-char hex, regenerated on every backend restart
_ingest_status: dict = {"state": "idle", "last_run": None, "error": None}
_ingest_lock = threading.Lock()

DOCUMENTS_DIR = Path("/app/documents")
CSV_PATH = Path("/app/logs/usage.csv")

ALLOWED_SUFFIXES = {".pdf", ".docx", ".txt"}
MAX_UPLOAD_BYTES = 20 * 1024 * 1024  # 20 MB

TOPICS = [
    ("PTO / Vacation", ["pto", "vacation", "time off", "days off", "accrual", "accrued"]),
    ("Benefits / Insurance", ["benefit", "insurance", "health", "dental", "vision", "medical", "coverage"]),
    ("FMLA / Leave", ["fmla", "leave", "maternity", "paternity", "family leave", "medical leave"]),
    ("Bereavement", ["bereavement", "funeral", "death", "passing"]),
    ("Holidays", ["holiday", "christmas", "thanksgiving", "labor day", "memorial day", "new year"]),
    ("Pay / Payroll", ["pay", "payroll", "salary", "wage", "overtime", "direct deposit"]),
    ("Conduct / Policy", ["conduct", "policy", "disciplinary", "harassment", "code of conduct"]),
    ("Other", []),
]

_security = HTTPBearer(auto_error=False)


def get_auth_token() -> str:
    """Exposed for tests only."""
    return _token


def require_auth(credentials: Optional[HTTPAuthorizationCredentials] = Depends(_security)) -> None:
    if credentials is None:
        raise HTTPException(status_code=403, detail="Missing credentials")
    if credentials.credentials != _token:
        raise HTTPException(status_code=401, detail="Invalid token")


class LoginRequest(BaseModel):
    password: str


@router.post("/login")
def login(req: LoginRequest):
    settings = get_settings()
    if not settings.admin_password:
        raise HTTPException(status_code=503, detail="Admin portal not configured.")
    if req.password != settings.admin_password:
        raise HTTPException(status_code=401, detail="Invalid password")
    return {"token": _token}


def classify_topic(question: str) -> str:
    q = question.lower()
    for topic, keywords in TOPICS[:-1]:  # skip Other (catch-all)
        if any(kw in q for kw in keywords):
            return topic
    return "Other"


def _read_csv_rows() -> list[dict]:
    if not CSV_PATH.exists():
        return []
    with open(CSV_PATH, "r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


@router.get("/logs")
def get_logs(_: None = Depends(require_auth)):
    return _read_csv_rows()


@router.get("/analytics")
def get_analytics(_: None = Depends(require_auth)):
    rows = _read_csv_rows()
    if not rows:
        topic_list = [{"topic": t, "count": 0} for t, _ in TOPICS]
        return {"totals": {"total": 0, "sessions": 0, "answered": 0, "redirected": 0}, "daily": [], "topics": topic_list}

    total = len(rows)
    sessions = len(set(r.get("Session ID", "") for r in rows if r.get("Session ID")))
    answered = sum(1 for r in rows if r.get("Outcome") == "answered")
    redirected = total - answered

    daily: dict[str, int] = defaultdict(int)
    for r in rows:
        ts = r.get("Timestamp", "")
        if ts:
            try:
                # Handle both ISO (2026-08-19) and M/D/YYYY (8/19/2026 12:08) formats
                if "/" in ts:
                    parsed = datetime.strptime(ts.strip(), "%m/%d/%Y %H:%M")
                else:
                    parsed = datetime.fromisoformat(ts[:16])
                daily[parsed.date().isoformat()] += 1
            except ValueError:
                pass

    today = date.today()
    daily_list = [
        {"date": (today - timedelta(days=i)).isoformat(), "count": daily.get((today - timedelta(days=i)).isoformat(), 0)}
        for i in range(29, -1, -1)
    ]

    topic_counts: dict[str, int] = defaultdict(int)
    for r in rows:
        topic_counts[classify_topic(r.get("Question", ""))] += 1

    lang_names = {"en": "English", "es": "Español", "ps": "پښتو", "fa": "دری", "bs": "Bosanski"}
    lang_counts: dict[str, int] = defaultdict(int)
    for r in rows:
        lang = (r.get("Language", "en") or "en").lower()
        lang_counts[lang] += 1
    languages = [{"code": c, "name": n, "count": lang_counts.get(c, 0)} for c, n in lang_names.items()]

    recent = [
        {
            "timestamp": r.get("Timestamp", ""),
            "question": r.get("Question", ""),
            "outcome": r.get("Outcome", ""),
            "language": r.get("Language", "en"),
        }
        for r in reversed(rows[-5:])
    ]

    return {
        "totals": {"total": total, "sessions": sessions, "answered": answered, "redirected": redirected},
        "daily": daily_list,
        "topics": [{"topic": t, "count": topic_counts.get(t, 0)} for t, _ in TOPICS],
        "languages": languages,
        "recent": recent,
    }


@router.get("/documents")
def list_documents(_: None = Depends(require_auth)):
    if not DOCUMENTS_DIR.exists():
        return []
    return [
        {
            "name": f.name,
            "size": f.stat().st_size,
            "modified": datetime.fromtimestamp(f.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
        }
        for f in sorted(DOCUMENTS_DIR.iterdir())
        if f.is_file() and f.suffix.lower() in ALLOWED_SUFFIXES
    ]


@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    _: None = Depends(require_auth),
):
    if Path(file.filename).suffix.lower() not in ALLOWED_SUFFIXES:
        raise HTTPException(status_code=400, detail="Only PDF, DOCX, and TXT files are allowed.")
    chunks = []
    size = 0
    while True:
        chunk = await file.read(65536)
        if not chunk:
            break
        size += len(chunk)
        if size > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=400, detail="File exceeds 20 MB limit.")
        chunks.append(chunk)
    content = b"".join(chunks)
    dest = DOCUMENTS_DIR / Path(file.filename).name
    dest.write_bytes(content)
    return {"filename": dest.name, "size": size}


@router.delete("/documents/{filename}")
def delete_document(filename: str, _: None = Depends(require_auth)):
    if not DOCUMENTS_DIR.exists():
        raise HTTPException(status_code=404, detail="Documents directory not found.")
    existing = [f for f in DOCUMENTS_DIR.iterdir() if f.is_file() and f.suffix.lower() in ALLOWED_SUFFIXES]
    if len(existing) <= 1:
        raise HTTPException(status_code=400, detail="Cannot delete the last document.")
    target = DOCUMENTS_DIR / filename
    # Guard against path traversal
    if not target.resolve().is_relative_to(DOCUMENTS_DIR.resolve()):
        raise HTTPException(status_code=400, detail="Invalid filename.")
    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail="Document not found.")
    target.unlink()
    return {"deleted": filename}


def _run_ingest_background() -> None:
    global _ingest_status
    try:
        from app.ingest import run_ingest
        settings = get_settings()
        run_ingest(qdrant_host=settings.qdrant_host)
        with _ingest_lock:
            _ingest_status = {"state": "done", "last_run": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "error": None}
    except Exception as exc:
        with _ingest_lock:
            _ingest_status = {"state": "error", "last_run": None, "error": str(exc)}


@router.post("/ingest")
def trigger_ingest(background_tasks: BackgroundTasks, _: None = Depends(require_auth)):
    global _ingest_status
    with _ingest_lock:
        if _ingest_status["state"] == "running":
            raise HTTPException(status_code=409, detail="Ingest already running.")
        _ingest_status = {"state": "running", "last_run": None, "error": None}
    background_tasks.add_task(_run_ingest_background)
    return {"status": "started"}


@router.get("/ingest/status")
def get_ingest_status(_: None = Depends(require_auth)):
    return _ingest_status
