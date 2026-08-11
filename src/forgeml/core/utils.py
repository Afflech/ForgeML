import time
from functools import wraps
from typing import Callable, Any

from forgeml.core.errors import ProviderError, AuthError, QuotaError
from forgeml.core.logging import get_logger

logger = get_logger(__name__)


def retry_transient(
    max_attempts: int = 3,
    initial_wait_s: int = 5,
    backoff_factor: float = 2.0,
) -> Callable:
    """
    Retry decorator for transient Kaggle API / Network errors.
    Will not retry AuthError or QuotaError.
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            attempt = 1
            wait_s = initial_wait_s
            while True:
                try:
                    return func(*args, **kwargs)
                except (AuthError, QuotaError) as e:
                    # Terminal provider errors, never retry
                    raise e
                except ProviderError as e:
                    if attempt >= max_attempts:
                        logger.error("Transient error %s failed after %d attempts", func.__name__, max_attempts)
                        raise e
                    logger.warning(
                        "Transient error in %s: %s. Retrying in %ds (Attempt %d/%d)",
                        func.__name__, str(e), wait_s, attempt, max_attempts
                    )
                    time.sleep(wait_s)
                    attempt += 1
                    wait_s = int(wait_s * backoff_factor)
                except Exception as e:
                    # Generic exceptions aren't caught by this retry (unless they are wrapped as ProviderError)
                    raise e
        return wrapper
    return decorator