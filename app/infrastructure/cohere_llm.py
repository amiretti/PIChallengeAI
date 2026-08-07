import cohere

from app.domain.errors import LanguageModelError


class CohereLanguageModel:
    def __init__(
        self, api_key: str, model: str, temperature: float = 0.0, seed: int = 42
    ) -> None:
        self._client = cohere.ClientV2(api_key=api_key)
        self._model = model
        self._temperature = temperature
        self._seed = seed

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        try:
            response = self._client.chat(
                model=self._model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self._temperature,
                seed=self._seed,
            )
        except Exception as error:
            raise LanguageModelError(f"Cohere request failed: {error}") from error

        return "".join(block.text for block in response.message.content).strip()
