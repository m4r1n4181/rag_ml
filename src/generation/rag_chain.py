from loguru import logger
from src.retrieval.vector_store import search, load_vector_store
from src.generation.prompts import build_prompt, SYSTEM_PROMPT
from src.generation.llm import generate, generate_stream
from dataclasses import dataclass


@dataclass
class RAGResponse:
    """
    UI needs sources and answer.
    """
    answer: str
    sources: list[dict]
    query: str


def ask(query: str, n_results: int = 5) -> RAGResponse:
    """
    testing in terminal
    """
    collection = load_vector_store()

    # Step 1: Retrieve
    logger.info(f"Pretražujem za: '{query}'")
    chunks = search(query, collection, n_results=n_results)

    if not chunks:
        return RAGResponse(
            answer="Nisam pronašao relevantne informacije u materijalima.",
            sources=[],
            query=query
        )

    logger.info(f"Pronađeno {len(chunks)} relevantnih chunkova")
    for i, c in enumerate(chunks, 1):
        logger.debug(f"  [{i}] similarity={c['similarity']} | {c['metadata'].get('source')}")

    # Step 2: Augment - context into prompt
    prompt = build_prompt(query, chunks)

    # Step 3: Generate
    logger.info("Generišem odgovor...")
    answer = generate(prompt, SYSTEM_PROMPT)

    return RAGResponse(
        answer=answer,
        sources=chunks,
        query=query
    )


def ask_stream(query: str, collection=None, n_results: int = 5):
    """
    Ako collection nije prosleđen, učitava ga sam.
    Streamlit ga prosleđuje iz session_state — učita se jednom.
    """
    if collection is None:
        collection = load_vector_store()

    chunks = search(query, collection, n_results=n_results)

    if not chunks:
        yield ("token", "Nisam pronašao relevantne informacije u materijalima.")
        yield ("sources", [])
        return

    prompt = build_prompt(query, chunks)

    for token in generate_stream(prompt, SYSTEM_PROMPT):
        yield ("token", token)

    yield ("sources", chunks)