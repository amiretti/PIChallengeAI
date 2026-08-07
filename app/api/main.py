import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api.routes import router
from app.application.answer_question import AnswerQuestion
from app.application.index_document import IndexDocument
from app.config import Settings
from app.domain.errors import RagError
from app.infrastructure.chroma_vector_store import ChromaVectorStore
from app.infrastructure.cohere_llm import CohereLanguageModel
from app.infrastructure.docx_loader import DocxDocumentLoader

logger = logging.getLogger(__name__)

# Matches uvicorn's own format, so application and server lines read as one log.
LOG_FORMAT = "%(levelname)s:     %(message)s"
UPSTREAM_ERROR_STATUS = 502
UPSTREAM_ERROR_MESSAGE = "The question could not be answered right now, please retry."


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Builds the adapters and indexes the document once, before the API serves traffic."""
    settings = Settings()

    vector_store = ChromaVectorStore(
        settings.cohere_api_key, settings.cohere_embedding_model
    )
    indexed = IndexDocument(DocxDocumentLoader(), vector_store).execute(
        settings.document_path
    )
    logger.info("Indexed %d chunks from %s", indexed, settings.document_path)

    language_model = CohereLanguageModel(
        settings.cohere_api_key,
        settings.cohere_chat_model,
        settings.llm_temperature,
        settings.llm_seed,
    )
    app.state.answer_question = AnswerQuestion(
        vector_store, language_model, settings.top_k
    )
    yield


async def handle_rag_error(request: Request, error: Exception) -> JSONResponse:
    """Turns any domain error into a generic 502, keeping provider details server-side."""
    # Cohere errors carry the request headers and account details: log them, never return them.
    logger.exception("Request to %s failed: %s", request.url.path, error)
    return JSONResponse(
        status_code=UPSTREAM_ERROR_STATUS, content={"detail": UPSTREAM_ERROR_MESSAGE}
    )


def create_app() -> FastAPI:
    # uvicorn configures its own loggers but leaves the root logger at WARNING, so without
    # this the application's INFO messages (the indexing summary) are silently dropped.
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)

    app = FastAPI(
        title="PI Challenge — RAG API",
        description=(
            "Answers questions about a document using retrieval augmented generation."
        ),
        version="1.0.0",
        lifespan=lifespan,
    )
    app.add_exception_handler(RagError, handle_rag_error)
    app.include_router(router)
    return app


app = create_app()
