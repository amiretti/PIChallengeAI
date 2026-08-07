from fastapi import Request

from app.application.answer_question import AnswerQuestion


def get_answer_question(request: Request) -> AnswerQuestion:
    """Returns the use case built during startup and kept on the application state."""
    return request.app.state.answer_question
