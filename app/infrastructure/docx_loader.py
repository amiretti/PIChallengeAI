from pathlib import Path

import docx
from docx.opc.exceptions import PackageNotFoundError

from app.domain.errors import DocumentNotFound, DocumentReadError


class DocxDocumentLoader:
    def load(self, path: str) -> str:
        source = Path(path)
        if not source.is_file():
            raise DocumentNotFound(f"Document not found: {source}")

        try:
            document = docx.Document(str(source))
        except PackageNotFoundError as error:
            raise DocumentReadError(f"Not a readable .docx file: {source}") from error

        return "\n".join(paragraph.text for paragraph in document.paragraphs)
