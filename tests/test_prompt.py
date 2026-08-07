from app.application.prompt import (
    EMPTY_CONTEXT,
    LANGUAGE_REMINDER,
    SYSTEM_PROMPT,
    build_user_prompt,
)
from app.domain.models import Chunk, RetrievedChunk


def retrieved(text: str, chunk_id: str = "chunk-0") -> RetrievedChunk:
    return RetrievedChunk(chunk=Chunk(id=chunk_id, text=text), distance=0.1)


def test_system_prompt_states_every_answer_requirement():
    prompt = SYSTEM_PROMPT.lower()

    assert "one sentence" in prompt
    assert "same language" in prompt
    assert "third person" in prompt
    assert "emoji" in prompt
    assert "context" in prompt


def test_user_prompt_contains_the_question_and_the_context():
    prompt = build_user_prompt("¿Quién es Zara?", [retrieved("Zara es una exploradora.")])

    assert "¿Quién es Zara?" in prompt
    assert "Zara es una exploradora." in prompt


def test_user_prompt_joins_every_retrieved_chunk():
    chunks = [retrieved("First fact.", "chunk-0"), retrieved("Second fact.", "chunk-1")]

    prompt = build_user_prompt("Any question?", chunks)

    assert "First fact." in prompt
    assert "Second fact." in prompt


def test_user_prompt_marks_the_context_as_empty_when_nothing_was_retrieved():
    prompt = build_user_prompt("Any question?", [])

    assert EMPTY_CONTEXT in prompt


def test_user_prompt_repeats_the_language_rule_after_the_question():
    # Without this reminder the model answers in the language of the context.
    prompt = build_user_prompt("Who is Zara?", [retrieved("Zara es una exploradora.")])

    assert prompt.index(LANGUAGE_REMINDER) > prompt.index("Who is Zara?")


def test_user_prompt_is_deterministic_for_the_same_inputs():
    chunks = [retrieved("Zara es una exploradora.")]

    assert build_user_prompt("¿Quién es Zara?", chunks) == build_user_prompt(
        "¿Quién es Zara?", chunks
    )
