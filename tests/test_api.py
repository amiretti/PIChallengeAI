import pytest
from fastapi.testclient import TestClient

from app.api.dependencies import get_answer_question
from app.api.main import create_app
from app.domain.errors import LanguageModelError, VectorStoreError
from app.domain.models import Answer, Question


class StubAnswerQuestion:
    def __init__(self, answer: str = "Zara es una exploradora. 🚀", error: Exception | None = None):
        self._answer = answer
        self._error = error
        self.questions: list[Question] = []

    def execute(self, question: Question) -> Answer:
        self.questions.append(question)
        if self._error is not None:
            raise self._error
        return Answer(text=self._answer)


def build_client(use_case: StubAnswerQuestion) -> TestClient:
    # No context manager: the lifespan would index the document against the real API.
    app = create_app()
    app.dependency_overrides[get_answer_question] = lambda: use_case
    return TestClient(app)


def test_returns_the_answer_for_a_valid_request():
    client = build_client(StubAnswerQuestion("Zara es una exploradora. 🚀"))

    response = client.post("/", json={"user_name": "John Doe", "question": "¿Quién es Zara?"})

    assert response.status_code == 200
    assert response.json() == {"answer": "Zara es una exploradora. 🚀"}


def test_forwards_the_user_name_and_question_to_the_use_case():
    use_case = StubAnswerQuestion()
    client = build_client(use_case)

    client.post("/", json={"user_name": "John Doe", "question": "¿Quién es Zara?"})

    assert use_case.questions == [Question(user_name="John Doe", text="¿Quién es Zara?")]


@pytest.mark.parametrize(
    "payload",
    [
        {"question": "¿Quién es Zara?"},
        {"user_name": "John Doe"},
        {"user_name": "John Doe", "question": ""},
        {"user_name": "John Doe", "question": "   "},
        {"user_name": "", "question": "¿Quién es Zara?"},
        {"user_name": "John Doe", "question": 42},
    ],
)
def test_rejects_invalid_requests(payload):
    client = build_client(StubAnswerQuestion())

    assert client.post("/", json=payload).status_code == 422


def test_trims_surrounding_whitespace_from_the_question():
    use_case = StubAnswerQuestion()
    client = build_client(use_case)

    client.post("/", json={"user_name": " John Doe ", "question": "  ¿Quién es Zara?  "})

    assert use_case.questions[0].text == "¿Quién es Zara?"


@pytest.mark.parametrize(
    "error", [LanguageModelError("cohere key sk-secret leaked"), VectorStoreError("boom")]
)
def test_returns_502_when_an_upstream_dependency_fails(error):
    client = build_client(StubAnswerQuestion(error=error))

    response = client.post("/", json={"user_name": "John Doe", "question": "¿Quién es Zara?"})

    assert response.status_code == 502


def test_error_response_does_not_leak_internal_details():
    # Cohere errors carry response headers and account details; they must not reach the client.
    client = build_client(StubAnswerQuestion(error=LanguageModelError("api key sk-secret")))

    response = client.post("/", json={"user_name": "John Doe", "question": "¿Quién es Zara?"})

    assert "sk-secret" not in response.text


def test_health_endpoint_reports_the_service_is_up():
    client = build_client(StubAnswerQuestion())

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
