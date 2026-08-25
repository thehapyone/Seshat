"""Service logger."""

import logging
import sys

logger = logging.getLogger("seshat")


def setup_logging(level: str = "INFO") -> None:
    """Configure a single stdout handler for the service logger."""
    resolved = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(resolved)
    if logger.handlers:
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    )
    logger.addHandler(handler)
    logger.propagate = False
