from typing import Annotated

from pydantic import BaseModel, Field, StringConstraints

MAX_USER_NAME_LENGTH = 100
MAX_QUESTION_LENGTH = 1000

# Trimmed first, then required to have content: a question of only spaces is not a question.
TrimmedText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class QuestionRequest(BaseModel):
    user_name: Annotated[
        TrimmedText, Field(max_length=MAX_USER_NAME_LENGTH, examples=["John Doe"])
    ]
    question: Annotated[
        TrimmedText, Field(max_length=MAX_QUESTION_LENGTH, examples=["¿Quién es Zara?"])
    ]


class AnswerResponse(BaseModel):
    answer: str


class HealthResponse(BaseModel):
    status: str
