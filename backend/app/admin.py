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


@router.get("/logs")
def get_logs(auth: None = Depends(require_auth)):
    return {"status": "ok"}
