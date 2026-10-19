import anthropic

from src.config import settings

_client = None


def get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    return _client


def _request(prompt: str, system_prompt: str | None, max_tokens: int | None) -> dict:
    kwargs = dict(
        model=settings.llm_model,
        max_tokens=max_tokens or settings.llm_max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    # Newer Claude models reject sampling parameters; set LLM_TEMPERATURE=none for those.
    if settings.llm_temperature is not None:
        kwargs["temperature"] = settings.llm_temperature
    if system_prompt:
        kwargs["system"] = system_prompt
    return kwargs


def generate(prompt: str, system_prompt: str | None = None, max_tokens: int | None = None) -> str:
    return get_client().messages.create(**_request(prompt, system_prompt, max_tokens)).content[0].text


def generate_stream(prompt: str, system_prompt: str):
    with get_client().messages.stream(**_request(prompt, system_prompt, None)) as stream:
        yield from stream.text_stream
