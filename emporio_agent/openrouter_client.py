"""The OpenRouter SDK client for the chat model and the embeddings, with its limits fixed here."""

from openrouter import OpenRouter
from openrouter.utils.logger import NoOpLogger
from openrouter.utils.retries import BackoffStrategy, RetryConfig

from emporio_agent.config import OPENROUTER_BASE_URL, Settings


def create_openrouter_client(settings: Settings, timeout_ms: int) -> OpenRouter:
    """Create an OpenRouter client; key, URL, time limit and a silent logger are set here."""
    return OpenRouter(
        api_key=settings.openrouter_api_key,
        server_url=OPENROUTER_BASE_URL,
        retry_config=single_attempt(),
        timeout_ms=timeout_ms,
        debug_logger=NoOpLogger(),
    )


def single_attempt() -> RetryConfig:
    """Never retry: the SDK's retries wait as long as a server's Retry-After header asks."""
    unused_backoff = BackoffStrategy(
        initial_interval=0, max_interval=0, exponent=1.0, max_elapsed_time=0
    )
    return RetryConfig("none", unused_backoff, retry_connection_errors=False)
