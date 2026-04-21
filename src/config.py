from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # LLM
    anthropic_api_key: str = ""
    llm_model: str = "claude-haiku-4-5-20251001"
    llm_temperature: float = 0.1
    llm_max_tokens: int = 1024

    # Embeddings
    embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2"

    # Reranking
    reranker_model: str = "BAAI/bge-reranker-v2-m3"

    # Vector Store
    chroma_path: str = "data/chroma_db"
    collection_name: str = "faculty_rag"
    batch_size: int = 64

    # RAG
    n_results: int = 5
    chunk_size: int = 800
    chunk_overlap: int = 250

settings = Settings()