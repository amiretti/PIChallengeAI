"""End-to-end checks of the RAG pipeline against the real Cohere API.

These verify the prompt itself: the answer requirements can only be proven with a real model.
"""

import re

import pytest

from app.application.answer_question import AnswerQuestion
from app.application.index_document import IndexDocument
from app.config import Settings
from app.domain.models import Question
from app.infrastructure.chroma_vector_store import ChromaVectorStore
from app.infrastructure.cohere_llm import CohereLanguageModel
from app.infrastructure.docx_loader import DocxDocumentLoader

pytestmark = pytest.mark.integration

EMOJI_PATTERN = re.compile(
    "[\U0001f300-\U0001faff\U00002600-\U000027bf\U0001f1e6-\U0001f1ff]"
)
FIRST_OR_SECOND_PERSON = re.compile(
    r"\b(I|me|my|you|your|we|our|yo|mi|mí|tú|tu|usted|vos|nosotros|eu|meu|você|nós)\b",
    re.IGNORECASE,
)


@pytest.fixture(scope="module")
def settings():
    try:
        return Settings()
    except Exception:
        pytest.skip("COHERE_API_KEY is not configured")


@pytest.fixture(scope="module")
def answer_question(settings):
    store = ChromaVectorStore(settings.cohere_api_key, settings.cohere_embedding_model)
    IndexDocument(DocxDocumentLoader(), store).execute(settings.document_path)
    model = CohereLanguageModel(
        settings.cohere_api_key,
        settings.cohere_chat_model,
        settings.llm_temperature,
        settings.llm_seed,
    )
    return AnswerQuestion(store, model, settings.top_k)


@pytest.fixture(scope="module")
def zara_answer(answer_question):
    return answer_question.execute(Question("John Doe", "¿Quién es Zara?")).text


def test_answers_the_question_from_the_document(zara_answer):
    assert "Zara" in zara_answer


def test_answers_with_a_single_sentence(zara_answer):
    # One terminator at most, and only at the end of the text.
    assert len(re.findall(r"[.!?…](?:\s|$)", zara_answer)) <= 1


def test_answer_includes_emojis(zara_answer):
    assert EMOJI_PATTERN.search(zara_answer)


def test_answer_stays_in_third_person(zara_answer):
    assert not FIRST_OR_SECOND_PERSON.search(zara_answer)


def test_same_question_always_gets_the_same_answer(answer_question, zara_answer):
    repeated = answer_question.execute(Question("Jane Roe", "¿Quién es Zara?")).text

    assert repeated == zara_answer


@pytest.mark.parametrize(
    "question, expected_text",
    [
        ("What did Emma decide to do?", "Emma"),
        ("What is the name of the magical flower?", "Luz de Luna"),
    ],
)
def test_answers_the_challenge_sample_questions(answer_question, question, expected_text):
    answer = answer_question.execute(Question("John Doe", question)).text

    assert expected_text in answer


# "the"/"is" have no Spanish homographs, and none of these Spanish markers appear inside
# the proper nouns of the document (Zara, Luz de Luna, Zenthoria, Dracorians, Lumis).
ENGLISH_MARKER = re.compile(r"\b(the|is|was|who)\b", re.IGNORECASE)
SPANISH_MARKER = re.compile(r"\b(es|una|un|que|los|las|del|se)\b", re.IGNORECASE)


@pytest.mark.parametrize(
    "question",
    [
        "Who is Zara?",
        "What did Emma decide to do?",
        "What is the name of the magical flower?",
    ],
)
def test_answers_english_questions_in_english(answer_question, question):
    # The document is in Spanish, so the model must follow the question, not the context.
    answer = answer_question.execute(Question("John Doe", question)).text

    assert ENGLISH_MARKER.search(answer)
    assert not SPANISH_MARKER.search(answer)


def test_answers_portuguese_questions_in_portuguese(answer_question):
    answer = answer_question.execute(Question("John Doe", "Quem são os Dracorians?")).text

    assert re.search(r"\b(são|uma|dos|na)\b", answer, re.IGNORECASE)
