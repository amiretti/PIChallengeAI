from app.application.prompt import SYSTEM_PROMPT, build_user_prompt
from app.domain.models import Answer, Question
from app.domain.ports import LanguageModel, VectorStore

DEFAULT_TOP_K = 1


def build_cache_key(question_text: str) -> str:
    """Normalises spacing and casing so the same question always hits the same entry."""
    return " ".join(question_text.split()).lower()


class AnswerQuestion:
    """Retrieves the most relevant excerpts and asks the model to answer from them."""

    def __init__(
        self,
        vector_store: VectorStore,
        language_model: LanguageModel,
        top_k: int = DEFAULT_TOP_K,
    ) -> None:
        self._vector_store = vector_store
        self._language_model = language_model
        self._top_k = top_k
        # Hosted models are not bit-for-bit reproducible even at temperature 0, so this
        # cache is what actually guarantees "same question, same answer". It also spares
        # an API call, which matters under Cohere's trial-key rate limit.
        self._answers: dict[str, Answer] = {}

    def execute(self, question: Question) -> Answer:
        # question.user_name is intentionally left out of the prompt and the cache key:
        # the same question must always produce the same answer, no matter who asks it.
        cache_key = build_cache_key(question.text)
        cached = self._answers.get(cache_key)
        if cached is not None:
            return cached

        retrieved = self._vector_store.search(question.text, self._top_k)
        user_prompt = build_user_prompt(question.text, retrieved)
        answer = Answer(text=self._language_model.generate(SYSTEM_PROMPT, user_prompt))

        self._answers[cache_key] = answer
        return answer
