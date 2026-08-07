from typing import Protocol, Sequence

from app.domain.models import Chunk, RetrievedChunk


class DocumentLoader(Protocol):
    def load(self, path: str) -> str:
        """Returns the document text with paragraphs separated by newlines."""
        ...


class VectorStore(Protocol):
    def add(self, chunks: Sequence[Chunk]) -> None: ...

    def search(self, query: str, top_k: int) -> list[RetrievedChunk]:
        """Returns the closest chunks, nearest first."""
        ...


class LanguageModel(Protocol):
    def generate(self, system_prompt: str, user_prompt: str) -> str: ...
