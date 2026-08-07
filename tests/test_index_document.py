from app.application.index_document import IndexDocument
from tests.fakes import FakeDocumentLoader, FakeVectorStore


def test_stores_one_chunk_per_paragraph_of_the_document():
    loader = FakeDocumentLoader("First story.\nSecond story.")
    store = FakeVectorStore()

    IndexDocument(loader, store).execute("data/documento.docx")

    assert [chunk.text for chunk in store.added_chunks] == [
        "First story.",
        "Second story.",
    ]


def test_reads_the_requested_path():
    loader = FakeDocumentLoader("Only story.")

    IndexDocument(loader, FakeVectorStore()).execute("data/other.docx")

    assert loader.loaded_paths == ["data/other.docx"]


def test_returns_the_number_of_indexed_chunks():
    loader = FakeDocumentLoader("One.\nTwo.\nThree.")

    indexed = IndexDocument(loader, FakeVectorStore()).execute("data/documento.docx")

    assert indexed == 3


def test_indexes_nothing_when_the_document_is_blank():
    store = FakeVectorStore()

    indexed = IndexDocument(FakeDocumentLoader("  \n\n "), store).execute("data/empty.docx")

    assert indexed == 0
    assert store.added_chunks == []
