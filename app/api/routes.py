from fastapi import APIRouter, Depends

from app.api.dependencies import get_answer_question
from app.api.schemas import AnswerResponse, HealthResponse, QuestionRequest
from app.application.answer_question import AnswerQuestion
from app.domain.models import Question

router = APIRouter()


@router.post("/", response_model=AnswerResponse, summary="Answer a question about the document")
def ask(
    payload: QuestionRequest,
    answer_question: AnswerQuestion = Depends(get_answer_question),
) -> AnswerResponse:
    # Sync on purpose: the Cohere and Chroma clients block, so FastAPI runs this handler in
    # a worker thread instead of stalling the event loop for every request.
    answer = answer_question.execute(
        Question(user_name=payload.user_name, text=payload.question)
    )
    return AnswerResponse(answer=answer.text)


@router.get("/health", response_model=HealthResponse, summary="Liveness probe")
def health() -> HealthResponse:
    return HealthResponse(status="ok")
