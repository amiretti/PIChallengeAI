from typing import Sequence

from app.domain.models import RetrievedChunk

# The first two rules are what tie the answer to the document, and they are the part the
# RESTRICT_TO_DOCUMENT setting swaps. Both variants contribute exactly two rules, so the
# numbering of the shared rules below stays the same in either mode.
DOCUMENT_ONLY_RULES = """1. Answer using only the facts present in the CONTEXT. Never add outside knowledge.
2. If the CONTEXT does not contain the answer, state that the document does not cover it."""

DOCUMENT_FIRST_RULES = """1. Prefer the facts present in the CONTEXT: it is the authoritative source, and it
   overrides anything you believe you know about the subject.
2. Only if the CONTEXT does not contain the answer, answer from your own knowledge."""

# Every answer requirement of the challenge is stated here as an explicit rule, except
# determinism, which no prompt can guarantee and AnswerQuestion enforces with a cache.
SYSTEM_PROMPT_TEMPLATE = """You answer questions about a document.

You receive a CONTEXT with excerpts of that document and a QUESTION about it.
Follow every rule below, without exception:

{grounding_rules}
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


def build_system_prompt(restrict_to_document: bool = True) -> str:
    """Renders the rules the model must follow.

    With `restrict_to_document` the model may only use the retrieved excerpts, which is
    the behaviour the challenge asks for. Without it, the excerpts still take precedence
    but the model may fall back to its own knowledge when the document is silent.
    """
    rules = DOCUMENT_ONLY_RULES if restrict_to_document else DOCUMENT_FIRST_RULES
    return SYSTEM_PROMPT_TEMPLATE.format(grounding_rules=rules)


def build_user_prompt(question: str, chunks: Sequence[RetrievedChunk]) -> str:
    """Renders the retrieved excerpts and the question into the prompt sent to the model."""
    context = CONTEXT_SEPARATOR.join(retrieved.chunk.text for retrieved in chunks)
    return USER_PROMPT_TEMPLATE.format(
        context=context or EMPTY_CONTEXT,
        question=question,
        language_reminder=LANGUAGE_REMINDER,
    )
