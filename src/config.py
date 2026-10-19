from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore", env_parse_none_str="none"
    )

    # LLM
    anthropic_api_key: str = ""
    llm_model: str = "claude-haiku-4-5-20251001"
    llm_temperature: float | None = 0.1
    llm_max_tokens: int = 1500

    # Embeddings (why e5 over MiniLM: see README, Results)
    embedding_model: str = "intfloat/multilingual-e5-large-instruct"
    query_instruction: str = (
        "Given a student question about machine learning course material, "
        "retrieve the lecture slides or textbook passages that answer it"
    )

    # Optional retrieval stages (both off by default, see data/eval/results.md)
    use_hybrid: bool = False
    use_reranker: bool = False
    reranker_model: str = "BAAI/bge-reranker-v2-m3"

    # Vector store
    chroma_path: str = "data/chroma_db"
    collection_name: str = "faculty_rag"
    batch_size: int = 32

    # Ingestion
    textbook_files: list[str] = ["MU_praktikum.pdf"]
    chunk_size: int = 800
    chunk_overlap: int = 150
    use_context_header: bool = True

    # Generation
    n_results: int = 10
    use_query_rewrite: bool = True

    # Public demo: caps API spending per browser session
    max_questions_per_session: int = 20


settings = Settings()
