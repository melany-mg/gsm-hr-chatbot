from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient

from app.config import Settings

SIMILARITY_THRESHOLD = 0.3

HR_REDIRECT = (
    "I'm sorry, I don't have information about that in the HR documents. "
    "Please contact HR directly: [HR_NAME], [HR_EMAIL], [HR_PHONE]."
)

LANGUAGE_NAMES = {
    "en": "English",
    "es": "Spanish",
    "ps": "Pashto",
    "fa": "Dari",
    "bs": "Bosnian",
}

SYSTEM_TEMPLATE = (
    "You are an HR assistant for General Stamping & Metalworks (GSM).\n"
    "Answer the employee's question using ONLY the context provided below.\n"
    "Do not use any knowledge outside of this context.\n"
    "If the context does not contain enough information to answer the question, "
    "say you don't know and direct the employee to HR.\n"
    "Respond in {language}.\n\n"
    "Context:\n{context}"
)


def build_llm(settings: Settings):
    if settings.environment == "development":
        from langchain_ollama import ChatOllama
        return ChatOllama(base_url=settings.ollama_base_url, model=settings.ollama_model)
    from langchain_anthropic import ChatAnthropic
    return ChatAnthropic(
        model=settings.claude_model,
        anthropic_api_key=settings.anthropic_api_key,
    )


def build_embeddings(settings: Settings) -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=settings.embedding_model)


def get_qdrant_client(settings: Settings) -> QdrantClient:
    return QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)


def retrieve_chunks(
    query: str,
    settings: Settings,
    embeddings: HuggingFaceEmbeddings,
    client: QdrantClient,
) -> list[tuple]:
    """Returns list of (Document, score) tuples, sorted by descending score."""
    store = QdrantVectorStore(
        client=client,
        collection_name=settings.qdrant_collection,
        embedding=embeddings,
    )
    return store.similarity_search_with_score(query, k=8)


def answer_question(
    message: str,
    language: str,
    history: list[dict],
    settings: Settings,
    embeddings: HuggingFaceEmbeddings,
    client: QdrantClient,
) -> dict:
    results = retrieve_chunks(message, settings, embeddings, client)
    relevant = [(doc, score) for doc, score in results if score >= SIMILARITY_THRESHOLD]

    if not relevant:
        return {"answer": HR_REDIRECT, "citations": []}

    context = "\n\n".join(doc.page_content for doc, _ in relevant)
    lang_name = LANGUAGE_NAMES.get(language, "English")
    system_prompt = SYSTEM_TEMPLATE.format(context=context, language=lang_name)

    # Build deduplicated citations from chunk metadata
    seen: set[tuple] = set()
    citations = []
    for doc, _ in relevant:
        key = (doc.metadata.get("source", ""), doc.metadata.get("page", 0))
        if key not in seen:
            seen.add(key)
            citations.append({"source": key[0], "page": key[1]})

    # Build message list for LLM
    messages = [SystemMessage(content=system_prompt)]
    for turn in history:
        if turn["role"] == "user":
            messages.append(HumanMessage(content=turn["content"]))
        else:
            messages.append(AIMessage(content=turn["content"]))
    messages.append(HumanMessage(content=message))

    llm = build_llm(settings)
    response = llm.invoke(messages)
    answer = response.content if hasattr(response, "content") else str(response)

    return {"answer": answer, "citations": citations}
