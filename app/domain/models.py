from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    id: str
    text: str


@dataclass(frozen=True)
class RetrievedChunk:
    chunk: Chunk
    distance: float


@dataclass(frozen=True)
class Question:
    user_name: str
    text: str


@dataclass(frozen=True)
class Answer:
    text: str
