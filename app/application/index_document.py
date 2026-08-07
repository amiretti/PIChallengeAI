from app.domain.chunking import split_into_chunks
from app.domain.ports import DocumentLoader, VectorStore


class IndexDocument:
    """Reads the source document, splits it into chunks and stores them in the vector store."""

    def __init__(self, loader: DocumentLoader, vector_store: VectorStore) -> None:
        self._loader = loader
        self._vector_store = vector_store

    def execute(self, document_path: str) -> int:
        """Indexes the document and returns how many chunks were stored."""
        chunks = split_into_chunks(self._loader.load(document_path))
        self._vector_store.add(chunks)
        return len(chunks)
