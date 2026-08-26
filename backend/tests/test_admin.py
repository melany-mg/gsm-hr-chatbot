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
