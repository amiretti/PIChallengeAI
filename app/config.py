from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    cohere_api_key: str
    cohere_embedding_model: str = "embed-multilingual-v3.0"
    # command-r/command-r-plus answer in the language of the context instead of the
    # language of the question; command-a follows the question. See README.
    cohere_chat_model: str = "command-a-03-2025"
    document_path: str = "data/documento.docx"
    top_k: int = 1
    llm_temperature: float = 0.0
    llm_seed: int = 42
    # True (the challenge behaviour): the model answers only from the document. False:
    # the document still wins, but the model may fall back to its own knowledge.
    restrict_to_document: bool = True
