import re
from dataclasses import dataclass

from loguru import logger

from src.config import settings
from src.generation.llm import generate, generate_stream
from src.generation.prompts import NO_ANSWER, QUERY_REWRITE_PROMPT, SYSTEM_PROMPT, build_prompt
from src.retrieval.vector_store import load_vector_store, search

# Words that usually mean the question refers back to the conversation
FOLLOW_UP_MARKERS = re.compile(
    r"^(a|i|pa|dakle)\b|\b(to|ovo|ono|taj|ta|tog|toga|tome|njega|nje|njih|isto|ovde|tamo|prethodn\w*|gore)\b",
    re.IGNORECASE,
)
MAX_REWRITE_CHARS = 300


@dataclass
class RAGResponse:
    answer: str
    sources: list[dict]
    query: str


def looks_like_follow_up(query: str) -> bool:
    return len(query.split()) <= 6 or bool(FOLLOW_UP_MARKERS.search(query))


def rewrite_query(query: str, history: list[dict] | None) -> str:
    """
    Turn a follow-up ("a kod Lasso?") into a standalone question so retrieval has
    something to work with. Skipped when there is no history or the question
    already looks self-contained, to avoid an extra LLM call on every message.
    """
    if not settings.use_query_rewrite or not history or not looks_like_follow_up(query):
        return query

    recent = "\n".join(
        f"{'Student' if m['role'] == 'user' else 'Asistent'}: {m['content'][:500]}" for m in history[-4:]
    )
    try:
        rewritten = generate(QUERY_REWRITE_PROMPT.format(history=recent, question=query), max_tokens=150).strip()
    except Exception as e:
        logger.error(f"Query rewrite failed, using the original question: {e}")
        return query

    # The model should return one short question; anything else is ignored.
    if not rewritten or "\n" in rewritten or len(rewritten) > MAX_REWRITE_CHARS:
        return query
    logger.info(f"Rewrote '{query}' -> '{rewritten}'")
    return rewritten


def ask(query: str, history: list[dict] | None = None, collection=None) -> RAGResponse:
    collection = collection or load_vector_store()
    search_query = rewrite_query(query, history)
    chunks = search(search_query, collection)
    if not chunks:
        return RAGResponse(answer=NO_ANSWER, sources=[], query=search_query)

    answer = generate(build_prompt(search_query, chunks), SYSTEM_PROMPT)
    return RAGResponse(answer=answer, sources=chunks, query=search_query)


def ask_stream(query: str, collection=None, history: list[dict] | None = None):
    """Yields ("query", str) once, ("token", str) while streaming, then ("sources", list)."""
    collection = collection or load_vector_store()
    search_query = rewrite_query(query, history)
    yield ("query", search_query)

    chunks = search(search_query, collection)
    if not chunks:
        yield ("token", NO_ANSWER)
        yield ("sources", [])
        return

    for token in generate_stream(build_prompt(search_query, chunks), SYSTEM_PROMPT):
        yield ("token", token)
    yield ("sources", chunks)
