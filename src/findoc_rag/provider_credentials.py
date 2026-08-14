import os
from collections.abc import Mapping
from urllib.parse import urlsplit

DEEPSEEK_API_HOST = "api.deepseek.com"
OPENAI_API_HOST = "api.openai.com"
CUSTOM_PROVIDER_API_KEY_ENV = "FINDOC_RAG_ANSWER_API_KEY"


def resolve_provider_api_key(
    endpoint: str,
    explicit_api_key: str | None = None,
    *,
    environ: Mapping[str, str] | None = None,
) -> str:
    """Resolve credentials only from the environment bound to the endpoint host."""
    if explicit_api_key is not None:
        return explicit_api_key

    environment = environ if environ is not None else os.environ
    host = (urlsplit(endpoint).hostname or "").lower()
    if host == DEEPSEEK_API_HOST:
        return environment.get("DEEPSEEK_API_KEY", "")
    if host == OPENAI_API_HOST:
        return environment.get("OPENAI_API_KEY", "")
    return environment.get(CUSTOM_PROVIDER_API_KEY_ENV, "")
