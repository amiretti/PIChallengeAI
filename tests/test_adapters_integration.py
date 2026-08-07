import pytest

from app.config import Settings
from app.domain.chunking import split_into_chunks
from app.infrastructure.chroma_vector_store import ChromaVectorStore
from app.infrastructure.cohere_llm import CohereLanguageModel
from app.infrastructure.docx_loader import DocxDocumentLoader

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def settings():
    try:
        return Settings()
    except Exception:
        pytest.skip("COHERE_API_KEY is not configured")


@pytest.fixture(scope="module")
def indexed_store(settings):
    store = ChromaVectorStore(settings.cohere_api_key, settings.cohere_embedding_model)
    store.add(split_into_chunks(DocxDocumentLoader().load(settings.document_path)))
    return store


@pytest.mark.parametrize(
    "question, expected_text",
    [
        ("¿Quién es Zara?", "Zara"),
        ("What did Emma decide to do?", "Emma decide compartir su regalo"),
        ("What is the name of the magical flower?", "Luz de Luna"),
        ("Quem são os Dracorians?", "Dracorians"),
    ],
)
def test_retrieves_the_relevant_chunk_across_languages(
    indexed_store, question, expected_text
):
    results = indexed_store.search(question, top_k=1)

    assert expected_text in results[0].chunk.text


def test_retrieval_is_deterministic(indexed_store):
    first = indexed_store.search("¿Quién es Zara?", top_k=1)
    second = indexed_store.search("¿Quién es Zara?", top_k=1)

    assert first[0].chunk == second[0].chunk


def test_reindexing_does_not_duplicate_chunks(settings, indexed_store):
    chunks = split_into_chunks(DocxDocumentLoader().load(settings.document_path))

    indexed_store.add(chunks)

    assert indexed_store._collection.count() == len(chunks)


def test_language_model_returns_text(settings):
    model = CohereLanguageModel(
        settings.cohere_api_key,
        settings.cohere_chat_model,
        settings.llm_temperature,
        settings.llm_seed,
    )

    answer = model.generate(
        system_prompt="Answer in one short sentence.",
        user_prompt="Who wrote Don Quixote?",
    )

    assert "Cervantes" in answer
