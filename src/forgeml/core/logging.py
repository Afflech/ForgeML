import logging
from rich.logging import RichHandler


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = RichHandler(
            rich_tracebacks=True,
            show_path=False,
            markup=True
        )
        handler.setFormatter(logging.Formatter("%(name)s — %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger
