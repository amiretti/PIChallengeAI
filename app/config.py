from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    cohere_api_key: str
    cohere_embedding_model: str = "embed-multilingual-v3.0"
    cohere_chat_model: str = "command-r-08-2024"
    document_path: str = "data/documento.docx"
    top_k: int = 1
    llm_temperature: float = 0.0
    llm_seed: int = 42
