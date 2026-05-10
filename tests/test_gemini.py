import os

import pytest


pytest.importorskip("langchain_google_genai")


def _require_gemini_integration() -> str:
    if os.getenv("RUN_GEMINI_INTEGRATION") != "1":
        pytest.skip("Set RUN_GEMINI_INTEGRATION=1 to run live Gemini API tests")

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        pytest.skip("GEMINI_API_KEY is not configured")

    return api_key


def test_gemini_embeddings_when_api_key_is_configured():
    api_key = _require_gemini_integration()

    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    embedder = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2", google_api_key=api_key)
    vector = embedder.embed_query("Hello world")

    assert isinstance(vector, list)
    assert len(vector) > 0
    assert all(isinstance(value, float) for value in vector)


def test_gemini_generation_when_api_key_is_configured():
    api_key = _require_gemini_integration()

    from langchain_google_genai import ChatGoogleGenerativeAI

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=api_key)
    response = llm.invoke("Reply with exactly: hi")

    assert str(response.content).strip()
