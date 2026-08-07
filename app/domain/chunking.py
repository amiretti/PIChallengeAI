from app.domain.models import Chunk

CHUNK_ID_PREFIX = "chunk"


def split_into_chunks(text: str) -> list[Chunk]:
    """Splits text into one chunk per paragraph, dropping blank lines."""
    paragraphs = [line.strip() for line in text.splitlines() if line.strip()]
    return [
        Chunk(id=f"{CHUNK_ID_PREFIX}-{index}", text=paragraph)
        for index, paragraph in enumerate(paragraphs)
    ]
