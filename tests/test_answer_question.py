from app.application.answer_question import AnswerQuestion
from app.application.prompt import build_system_prompt
from app.domain.models import Answer, Chunk, Question, RetrievedChunk
from tests.fakes import FakeLanguageModel, FakeVectorStore


def retrieved(text: str, chunk_id: str = "chunk-0") -> RetrievedChunk:
    return RetrievedChunk(chunk=Chunk(id=chunk_id, text=text), distance=0.1)


def test_returns_the_answer_produced_by_the_language_model():
    store = FakeVectorStore([retrieved("Zara es una exploradora.")])
    model = FakeLanguageModel("Zara es una exploradora espacial. 🚀")

    answer = AnswerQuestion(store, model).execute(Question("John Doe", "¿Quién es Zara?"))

    assert answer == Answer(text="Zara es una exploradora espacial. 🚀")


def test_searches_the_vector_store_with_the_question_text_and_top_k():
    store = FakeVectorStore([retrieved("Zara es una exploradora.")])

    AnswerQuestion(store, FakeLanguageModel(), top_k=2).execute(
        Question("John Doe", "¿Quién es Zara?")
    )

    assert store.searches == [("¿Quién es Zara?", 2)]


def test_passes_the_retrieved_context_and_the_question_to_the_language_model():
    store = FakeVectorStore([retrieved("Zara es una exploradora.")])
    model = FakeLanguageModel()

    AnswerQuestion(store, model).execute(Question("John Doe", "¿Quién es Zara?"))

    system_prompt, user_prompt = model.calls[0]
    assert system_prompt == build_system_prompt(restrict_to_document=True)
    assert "Zara es una exploradora." in user_prompt
    assert "¿Quién es Zara?" in user_prompt


def test_restricts_answers_to_the_document_by_default():
    model = FakeLanguageModel()

    AnswerQuestion(FakeVectorStore([retrieved("Zara.")]), model).execute(
        Question("John Doe", "¿Quién es Zara?")
    )

    assert model.calls[0][0] == build_system_prompt(restrict_to_document=True)


def test_uses_the_unrestricted_prompt_when_configured():
    model = FakeLanguageModel()

    AnswerQuestion(
        FakeVectorStore([retrieved("Zara.")]), model, restrict_to_document=False
    ).execute(Question("John Doe", "¿Quién ganó el mundial 2022?"))

    assert model.calls[0][0] == build_system_prompt(restrict_to_document=False)


def test_does_not_leak_the_user_name_into_the_prompt():
    # The same question must always produce the same answer, whoever asks it.
    model = FakeLanguageModel()

    AnswerQuestion(FakeVectorStore([retrieved("Zara.")]), model).execute(
        Question("John Doe", "¿Quién es Zara?")
    )

    assert "John Doe" not in "".join(model.calls[0])


def test_repeating_a_question_returns_the_first_answer_without_calling_the_model():
    # Hosted models are not bit-for-bit deterministic even at temperature 0, so the
    # cache is what actually guarantees "same question, same answer".
    store = FakeVectorStore([retrieved("Zara es una exploradora.")])
    model = FakeLanguageModel("Zara es una exploradora espacial. 🚀")
    use_case = AnswerQuestion(store, model)

    first = use_case.execute(Question("John Doe", "¿Quién es Zara?"))
    second = use_case.execute(Question("Jane Roe", "¿Quién es Zara?"))

    assert first == second
    assert len(model.calls) == 1


def test_cache_ignores_casing_and_surrounding_whitespace():
    model = FakeLanguageModel()
    use_case = AnswerQuestion(FakeVectorStore([retrieved("Zara.")]), model)

    use_case.execute(Question("John Doe", "¿Quién es Zara?"))
    use_case.execute(Question("John Doe", "  ¿QUIÉN es    Zara?  "))

    assert len(model.calls) == 1


def test_different_questions_get_their_own_answer():
    model = FakeLanguageModel()
    use_case = AnswerQuestion(FakeVectorStore([retrieved("Zara.")]), model)

    use_case.execute(Question("John Doe", "¿Quién es Zara?"))
    use_case.execute(Question("John Doe", "Who is Zara?"))

    assert len(model.calls) == 2


def test_still_asks_the_language_model_when_nothing_is_retrieved():
    model = FakeLanguageModel("El documento no menciona eso. 🤷")

    answer = AnswerQuestion(FakeVectorStore([]), model).execute(
        Question("John Doe", "¿Quién ganó el mundial?")
    )

    assert answer.text == "El documento no menciona eso. 🤷"
    assert len(model.calls) == 1
