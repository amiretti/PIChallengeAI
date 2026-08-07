"""In-memory test doubles for the domain ports, so use cases run without network."""

from typing import Sequence

from app.domain.models import Chunk, RetrievedChunk


class FakeDocumentLoader:
    def __init__(self, text: str) -> None:
        self._text = text
        self.loaded_paths: list[str] = []

    def load(self, path: str) -> str:
        self.loaded_paths.append(path)
        return self._text


class FakeVectorStore:
    def __init__(self, results: list[RetrievedChunk] | None = None) -> None:
        self._results = results or []
        self.added_chunks: list[Chunk] = []
        self.searches: list[tuple[str, int]] = []

    def add(self, chunks: Sequence[Chunk]) -> None:
        self.added_chunks.extend(chunks)

    def search(self, query: str, top_k: int) -> list[RetrievedChunk]:
        self.searches.append((query, top_k))
        return self._results[:top_k]


class FakeLanguageModel:
    def __init__(self, answer: str = "An answer. 🚀") -> None:
        self._answer = answer
        self.calls: list[tuple[str, str]] = []

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        self.calls.append((system_prompt, user_prompt))
        return self._answer
