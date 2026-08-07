from typing import Sequence

from app.domain.models import RetrievedChunk

# Every answer requirement of the challenge is stated here as an explicit rule, except
# determinism, which no prompt can guarantee and AnswerQuestion enforces with a cache.
SYSTEM_PROMPT = """You answer questions about a document.

You receive a CONTEXT with excerpts of that document and a QUESTION about it.
Follow every rule below, without exception:

1. Answer using only the facts present in the CONTEXT. Never add outside knowledge.
2. If the CONTEXT does not contain the answer, state that the document does not cover it.
3. Answer with exactly one sentence.
4. Write the answer in the same language as the QUESTION, even when the CONTEXT
   is written in a different language. Translate the facts if you need to.
5. Always write in the third person. Never use first or second person.
6. End the sentence with one to three emojis that summarise its content.
7. Output only that sentence: no greetings, no preamble, no explanations, and do not
   wrap it in quotation marks."""

CONTEXT_SEPARATOR = "\n\n"
EMPTY_CONTEXT = "(no excerpt was retrieved)"

# The reminder is repeated after the question because the document is in Spanish and the
# model tends to follow the language of the context instead of the language of the question.
LANGUAGE_REMINDER = (
    "Reply in the language of the QUESTION above, not in the language of the CONTEXT."
)

USER_PROMPT_TEMPLATE = """CONTEXT:
{context}

QUESTION:
{question}

{language_reminder}"""


def build_user_prompt(question: str, chunks: Sequence[RetrievedChunk]) -> str:
    """Renders the retrieved excerpts and the question into the prompt sent to the model."""
    context = CONTEXT_SEPARATOR.join(retrieved.chunk.text for retrieved in chunks)
    return USER_PROMPT_TEMPLATE.format(
        context=context or EMPTY_CONTEXT,
        question=question,
        language_reminder=LANGUAGE_REMINDER,
    )
