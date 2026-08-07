from typing import Sequence

import chromadb
from chromadb.utils import embedding_functions

from app.domain.errors import VectorStoreError
from app.domain.models import Chunk, RetrievedChunk

COLLECTION_NAME = "documents"


class ChromaVectorStore:
    def __init__(self, api_key: str, embedding_model: str) -> None:
        embedding_function = embedding_functions.CohereEmbeddingFunction(
            api_key=api_key, model_name=embedding_model
        )
        self._collection = chromadb.Client().get_or_create_collection(
            name=COLLECTION_NAME,
            configuration={"hnsw": {"space": "cosine"}},
            embedding_function=embedding_function,
        )

    def add(self, chunks: Sequence[Chunk]) -> None:
        if not chunks:
            return
        try:
            # upsert keeps re-indexing idempotent, since chunk ids are deterministic
            self._collection.upsert(
                ids=[chunk.id for chunk in chunks],
                documents=[chunk.text for chunk in chunks],
            )
        except Exception as error:
            raise VectorStoreError(f"Could not index chunks: {error}") from error

    def search(self, query: str, top_k: int) -> list[RetrievedChunk]:
        try:
            result = self._collection.query(query_texts=[query], n_results=top_k)
        except Exception as error:
            raise VectorStoreError(f"Could not query the vector store: {error}") from error

        return [
            RetrievedChunk(chunk=Chunk(id=chunk_id, text=text), distance=distance)
            for chunk_id, text, distance in zip(
                result["ids"][0], result["documents"][0], result["distances"][0]
            )
        ]
