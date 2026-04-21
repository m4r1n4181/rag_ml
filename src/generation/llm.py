import anthropic
from loguru import logger
from src.config import settings

# singleton pattern kao kod embeddera
# Anthropic klijent interno upravljа connection poolingom
_client = None


def get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    return _client


def generate(prompt: str, system_prompt: str) -> str:
    """For testing purposes."""
    client = get_client()

    message = client.messages.create(
        model=settings.llm_model,
        max_tokens=settings.llm_max_tokens,
        system=system_prompt,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    return message.content[0].text


def generate_stream(prompt: str, system_prompt: str):
    """
    Streaming version for Streamlit
    """
    client = get_client()

    with client.messages.stream(
            model=settings.llm_model,
            max_tokens=settings.llm_max_tokens,
            system=system_prompt,
            messages=[
                {"role": "user", "content": prompt}
            ]
    ) as stream:
        for text in stream.text_stream:
            yield text