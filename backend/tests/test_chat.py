from unittest.mock import MagicMock, patch
from langchain_core.documents import Document
from app.chat import answer_question, HR_REDIRECT, SIMILARITY_THRESHOLD
from app.config import Settings


def make_settings(**kwargs):
    defaults = dict(
        environment="development",
        ollama_base_url="http://localhost:11434",
        ollama_model="qwen2.5:7b",
        anthropic_api_key="",
        claude_model="claude-sonnet-4-6",
        qdrant_host="localhost",
        qdrant_port=6333,
        qdrant_collection="hr_documents",
        qdrant_external_host="localhost",
        embedding_model="paraphrase-multilingual-mpnet-base-v2",
    )
    defaults.update(kwargs)
    return Settings(**defaults)


def make_doc(content, source="handbook.pdf", page=1):
    return Document(page_content=content, metadata={"source": source, "page": page})


def test_returns_answer_when_context_found():
    settings = make_settings()
    mock_embeddings = MagicMock()
    mock_client = MagicMock()
    doc = make_doc("Employees receive 10 vacation days per year.")

    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="You receive 10 vacation days.")

    with patch("app.chat.retrieve_chunks", return_value=[(doc, 0.85)]), \
         patch("app.chat.build_llm", return_value=mock_llm):
        result = answer_question("How many vacation days?", "en", [], settings, mock_embeddings, mock_client)

    assert result["answer"] == "You receive 10 vacation days."
    assert result["citations"] == [{"source": "handbook.pdf", "page": 1}]


def test_returns_hr_redirect_when_below_threshold():
    settings = make_settings()
    mock_embeddings = MagicMock()
    mock_client = MagicMock()
    doc = make_doc("Irrelevant content.")

    with patch("app.chat.retrieve_chunks", return_value=[(doc, SIMILARITY_THRESHOLD - 0.01)]):
        result = answer_question("What is the weather today?", "en", [], settings, mock_embeddings, mock_client)

    assert result["answer"] == HR_REDIRECT
    assert result["citations"] == []


def test_returns_hr_redirect_when_no_chunks():
    settings = make_settings()
    mock_embeddings = MagicMock()
    mock_client = MagicMock()

    with patch("app.chat.retrieve_chunks", return_value=[]):
        result = answer_question("Random question", "en", [], settings, mock_embeddings, mock_client)

    assert result["answer"] == HR_REDIRECT
    assert result["citations"] == []


def test_deduplicates_citations():
    settings = make_settings()
    mock_embeddings = MagicMock()
    mock_client = MagicMock()
    doc1 = make_doc("Benefits info.", source="benefits.pdf", page=3)
    doc2 = make_doc("More benefits.", source="benefits.pdf", page=3)

    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="Here are your benefits.")

    with patch("app.chat.retrieve_chunks", return_value=[(doc1, 0.9), (doc2, 0.8)]), \
         patch("app.chat.build_llm", return_value=mock_llm):
        result = answer_question("What are my benefits?", "en", [], settings, mock_embeddings, mock_client)

    assert len(result["citations"]) == 1
    assert result["citations"][0] == {"source": "benefits.pdf", "page": 3}


def test_includes_conversation_history_in_llm_call():
    settings = make_settings()
    mock_embeddings = MagicMock()
    mock_client = MagicMock()
    doc = make_doc("FMLA allows 12 weeks of leave.")

    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="FMLA gives you 12 weeks.")

    history = [
        {"role": "user", "content": "What is FMLA?"},
        {"role": "assistant", "content": "FMLA is the Family and Medical Leave Act."},
    ]

    with patch("app.chat.retrieve_chunks", return_value=[(doc, 0.9)]), \
         patch("app.chat.build_llm", return_value=mock_llm):
        answer_question("How many weeks does it cover?", "en", history, settings, mock_embeddings, mock_client)

    call_args = mock_llm.invoke.call_args[0][0]
    # Should have system + 2 history messages + new user message = 4
    assert len(call_args) == 4


def test_uses_spanish_in_system_prompt():
    settings = make_settings()
    mock_embeddings = MagicMock()
    mock_client = MagicMock()
    doc = make_doc("Employees receive 10 vacation days.")

    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="Tienes 10 días de vacaciones.")

    with patch("app.chat.retrieve_chunks", return_value=[(doc, 0.9)]), \
         patch("app.chat.build_llm", return_value=mock_llm):
        answer_question("¿Cuántos días de vacaciones?", "es", [], settings, mock_embeddings, mock_client)

    system_msg = mock_llm.invoke.call_args[0][0][0]
    assert "Spanish" in system_msg.content
