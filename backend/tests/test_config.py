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
