import os
os.environ["ADMIN_PASSWORD"] = "testpass"  # must be set before app import

import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.config import get_settings

@pytest.fixture(autouse=True)
def reset_settings_cache():
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()

client = TestClient(app)

def get_token():
    res = client.post("/api/admin/login", json={"password": "testpass"})
    assert res.status_code == 200
    return res.json()["token"]

def test_login_wrong_password():
    res = client.post("/api/admin/login", json={"password": "wrong"})
    assert res.status_code == 401

def test_login_correct_password():
    res = client.post("/api/admin/login", json={"password": "testpass"})
    assert res.status_code == 200
    assert "token" in res.json()
    assert len(res.json()["token"]) == 48

def test_protected_endpoint_no_token():
    res = client.get("/api/admin/logs")
    assert res.status_code == 403

def test_protected_endpoint_wrong_token():
    res = client.get("/api/admin/logs", headers={"Authorization": "Bearer wrongtoken"})
    assert res.status_code == 401


import tempfile, os
from pathlib import Path

def test_get_logs_empty(tmp_path, monkeypatch):
    monkeypatch.setattr("app.admin.CSV_PATH", tmp_path / "nonexistent.csv")
    token = get_token()
    res = client.get("/api/admin/logs", headers={"Authorization": f"Bearer {token}"})
    assert res.status_code == 200
    assert res.json() == []

def test_get_logs_returns_rows(tmp_path, monkeypatch):
    csv_file = tmp_path / "usage.csv"
    csv_file.write_text(
        "Timestamp,Session ID,Language,Question,Answer,Outcome,Citations\n"
        "2026-08-26 09:00:00,abc123,en,How many vacation days?,10 days.,answered,5\n"
    )
    monkeypatch.setattr("app.admin.CSV_PATH", csv_file)
    token = get_token()
    res = client.get("/api/admin/logs", headers={"Authorization": f"Bearer {token}"})
    assert res.status_code == 200
    rows = res.json()
    assert len(rows) == 1
    assert rows[0]["Question"] == "How many vacation days?"

def test_get_analytics_empty(tmp_path, monkeypatch):
    monkeypatch.setattr("app.admin.CSV_PATH", tmp_path / "nonexistent.csv")
    token = get_token()
    res = client.get("/api/admin/analytics", headers={"Authorization": f"Bearer {token}"})
    assert res.status_code == 200
    data = res.json()
    assert data["totals"]["total"] == 0
    assert data["daily"] == []
    assert len(data["topics"]) == len(TOPICS_FOR_TEST)

TOPICS_FOR_TEST = [
    "PTO / Vacation", "Benefits / Insurance", "FMLA / Leave", "Bereavement",
    "Holidays", "Pay / Payroll", "Conduct / Policy", "Other"
]

def test_analytics_topic_classification(tmp_path, monkeypatch):
    csv_file = tmp_path / "usage.csv"
    csv_file.write_text(
        "Timestamp,Session ID,Language,Question,Answer,Outcome,Citations\n"
        "2026-08-26 09:00:00,abc,en,How much vacation do I get?,10 days.,answered,5\n"
        "2026-08-26 09:05:00,def,en,What is the holiday schedule?,Check handbook.,answered,3\n"
        "2026-08-26 09:10:00,ghi,en,Tell me about overtime pay?,Ask HR.,redirected,0\n"
    )
    monkeypatch.setattr("app.admin.CSV_PATH", csv_file)
    token = get_token()
    res = client.get("/api/admin/analytics", headers={"Authorization": f"Bearer {token}"})
    assert res.status_code == 200
    topics = {t["topic"]: t["count"] for t in res.json()["topics"]}
    assert topics["PTO / Vacation"] == 1
    assert topics["Holidays"] == 1
    assert topics["Pay / Payroll"] == 1


import io

def test_list_documents(tmp_path, monkeypatch):
    (tmp_path / "test.pdf").write_bytes(b"%PDF fake")
    (tmp_path / "notes.txt").write_text("hello")
    (tmp_path / "ignore.xyz").write_text("skip me")
    monkeypatch.setattr("app.admin.DOCUMENTS_DIR", tmp_path)
    token = get_token()
    res = client.get("/api/admin/documents", headers={"Authorization": f"Bearer {token}"})
    assert res.status_code == 200
    names = [f["name"] for f in res.json()]
    assert "test.pdf" in names
    assert "notes.txt" in names
    assert "ignore.xyz" not in names

def test_upload_pdf(tmp_path, monkeypatch):
    monkeypatch.setattr("app.admin.DOCUMENTS_DIR", tmp_path)
    token = get_token()
    res = client.post(
        "/api/admin/upload",
        headers={"Authorization": f"Bearer {token}"},
        files={"file": ("policy.pdf", b"%PDF content", "application/pdf")},
    )
    assert res.status_code == 200
    assert (tmp_path / "policy.pdf").exists()

def test_upload_invalid_type(tmp_path, monkeypatch):
    monkeypatch.setattr("app.admin.DOCUMENTS_DIR", tmp_path)
    token = get_token()
    res = client.post(
        "/api/admin/upload",
        headers={"Authorization": f"Bearer {token}"},
        files={"file": ("virus.exe", b"bad", "application/octet-stream")},
    )
    assert res.status_code == 400

def test_delete_document(tmp_path, monkeypatch):
    (tmp_path / "old.pdf").write_bytes(b"%PDF fake")
    (tmp_path / "keep.pdf").write_bytes(b"%PDF keep")
    monkeypatch.setattr("app.admin.DOCUMENTS_DIR", tmp_path)
    token = get_token()
    res = client.delete(
        "/api/admin/documents/old.pdf",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert res.status_code == 200
    assert not (tmp_path / "old.pdf").exists()

def test_delete_last_document_blocked(tmp_path, monkeypatch):
    (tmp_path / "only.pdf").write_bytes(b"%PDF only")
    monkeypatch.setattr("app.admin.DOCUMENTS_DIR", tmp_path)
    token = get_token()
    res = client.delete(
        "/api/admin/documents/only.pdf",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert res.status_code == 400


from unittest.mock import patch

def test_ingest_status_default():
    token = get_token()
    res = client.get("/api/admin/ingest/status", headers={"Authorization": f"Bearer {token}"})
    assert res.status_code == 200
    assert res.json()["state"] in ("idle", "done", "error")

def test_trigger_ingest(monkeypatch):
    token = get_token()
    with patch("app.admin._run_ingest_background") as mock_ingest:
        res = client.post("/api/admin/ingest", headers={"Authorization": f"Bearer {token}"})
    assert res.status_code == 200
    assert res.json()["status"] == "started"
