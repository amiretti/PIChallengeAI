import docx
import pytest

from app.domain.chunking import split_into_chunks
from app.domain.errors import DocumentNotFound, DocumentReadError
from app.infrastructure.docx_loader import DocxDocumentLoader

SOURCE_DOCUMENT = "data/documento.docx"


def make_docx(directory, paragraphs):
    path = directory / "sample.docx"
    document = docx.Document()
    for paragraph in paragraphs:
        document.add_paragraph(paragraph)
    document.save(path)
    return str(path)


def test_preserves_paragraphs_as_newlines(tmp_path):
    path = make_docx(tmp_path, ["First story.", "Second story."])

    assert DocxDocumentLoader().load(path) == "First story.\nSecond story."


def test_raises_when_file_is_missing(tmp_path):
    with pytest.raises(DocumentNotFound):
        DocxDocumentLoader().load(str(tmp_path / "missing.docx"))


def test_raises_when_file_is_not_a_docx(tmp_path):
    path = tmp_path / "fake.docx"
    path.write_text("this is not a docx")

    with pytest.raises(DocumentReadError):
        DocxDocumentLoader().load(str(path))


def test_source_document_yields_five_chunks():
    chunks = split_into_chunks(DocxDocumentLoader().load(SOURCE_DOCUMENT))

    assert len(chunks) == 5
    assert chunks[0].text.startswith("Ficción Espacial:")
    assert "Luz de Luna" in chunks[2].text
    assert "Emma decide compartir su regalo" in chunks[3].text
