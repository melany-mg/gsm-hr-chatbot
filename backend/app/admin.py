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

DOCUMENTS_DIR = Path("/app/documents")
CSV_PATH = Path("/app/logs/usage.csv")

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
            daily[ts[:10]] += 1

    today = date.today()
    daily_list = [
        {"date": (today - timedelta(days=i)).isoformat(), "count": daily.get((today - timedelta(days=i)).isoformat(), 0)}
        for i in range(29, -1, -1)
    ]

    topic_counts: dict[str, int] = defaultdict(int)
    for r in rows:
        topic_counts[classify_topic(r.get("Question", ""))] += 1

    return {
        "totals": {"total": total, "sessions": sessions, "answered": answered, "redirected": redirected},
        "daily": daily_list,
        "topics": [{"topic": t, "count": topic_counts.get(t, 0)} for t, _ in TOPICS],
    }
