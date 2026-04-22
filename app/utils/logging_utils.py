import logging
import time
from functools import wraps

from app.config import LOG_LEVEL


def setup_logging():
    logging.basicConfig(
        level=LOG_LEVEL,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def timed(logger, label):
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            start = time.time()
            try:
                return fn(*args, **kwargs)
            finally:
                elapsed = (time.time() - start) * 1000
                logger.info("%s took %.2f ms", label, elapsed)
        return wrapper
    return decorator
