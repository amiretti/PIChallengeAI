import pytest

from app.application.prompt import (
    EMPTY_CONTEXT,
    LANGUAGE_REMINDER,
    build_system_prompt,
    build_user_prompt,
)
from app.domain.models import Chunk, RetrievedChunk


def retrieved(text: str, chunk_id: str = "chunk-0") -> RetrievedChunk:
    return RetrievedChunk(chunk=Chunk(id=chunk_id, text=text), distance=0.1)


@pytest.mark.parametrize("restrict_to_document", [True, False])
def test_system_prompt_states_every_answer_requirement(restrict_to_document):
    # The answer requirements hold in both modes; only the grounding rules differ.
    prompt = build_system_prompt(restrict_to_document).lower()

    assert "one sentence" in prompt
    assert "same language" in prompt
    assert "third person" in prompt
    assert "emoji" in prompt
    assert "context" in prompt


def test_restricted_prompt_forbids_knowledge_outside_the_document():
    prompt = build_system_prompt(restrict_to_document=True).lower()

    assert "only the facts present in the context" in prompt
    assert "does not cover" in prompt


def test_unrestricted_prompt_allows_falling_back_to_model_knowledge():
    prompt = build_system_prompt(restrict_to_document=False).lower()

    assert "your own knowledge" in prompt
    assert "only the facts present in the context" not in prompt


def test_both_modes_keep_the_same_rule_numbering():
    # Each variant contributes exactly two rules, so rules 3-7 mean the same in both.
    restricted = build_system_prompt(restrict_to_document=True)
    unrestricted = build_system_prompt(restrict_to_document=False)

    for rule in ("3.", "4.", "5.", "6.", "7."):
        assert rule in restricted
        assert rule in unrestricted
    assert "8." not in restricted
    assert "8." not in unrestricted


def test_the_two_modes_produce_different_prompts():
    assert build_system_prompt(True) != build_system_prompt(False)


def test_defaults_to_restricting_answers_to_the_document():
    assert build_system_prompt() == build_system_prompt(restrict_to_document=True)


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
