from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient


def test_health():
    from app.main import app
    client = TestClient(app)
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_chat_returns_answer_and_citations():
    mock_result = {
        "answer": "You receive 10 vacation days.",
        "citations": [{"source": "handbook.pdf", "page": 4}],
    }
    with patch("app.main.answer_question", return_value=mock_result), \
         patch("app.main.get_embeddings", return_value=MagicMock()), \
         patch("app.main.get_client", return_value=MagicMock()):
        from app.main import app
        client = TestClient(app)
        response = client.post("/api/chat", json={
            "message": "How many vacation days?",
            "language": "en",
            "history": [],
        })

    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "You receive 10 vacation days."
    assert data["citations"][0]["source"] == "handbook.pdf"
    assert data["citations"][0]["page"] == 4


def test_chat_defaults_language_to_en():
    mock_result = {"answer": "Contact HR.", "citations": []}
    with patch("app.main.answer_question", return_value=mock_result) as mock_fn, \
         patch("app.main.get_embeddings", return_value=MagicMock()), \
         patch("app.main.get_client", return_value=MagicMock()):
        from app.main import app
        client = TestClient(app)
        client.post("/api/chat", json={"message": "What is FMLA?"})

    call_kwargs = mock_fn.call_args.kwargs
    assert call_kwargs["language"] == "en"


def test_chat_passes_history():
    mock_result = {"answer": "It covers 12 weeks.", "citations": []}
    history = [
        {"role": "user", "content": "What is FMLA?"},
        {"role": "assistant", "content": "It is the Family Medical Leave Act."},
    ]
    with patch("app.main.answer_question", return_value=mock_result) as mock_fn, \
         patch("app.main.get_embeddings", return_value=MagicMock()), \
         patch("app.main.get_client", return_value=MagicMock()):
        from app.main import app
        client = TestClient(app)
        client.post("/api/chat", json={
            "message": "How long does it last?",
            "language": "en",
            "history": history,
        })

    call_kwargs = mock_fn.call_args.kwargs
    assert len(call_kwargs["history"]) == 2
