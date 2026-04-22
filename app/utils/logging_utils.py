import logging
import time
from functools import wraps

from app.config import LOG_LEVEL


def setup_logging():
    logging.basicConfig(
        level=LOG_LEVEL,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def get_logger(name: str):
    return logging.getLogger(name)


def timed(logger, label, trace_id: str | None = None):
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            start = time.time()
            try:
                return fn(*args, **kwargs)
            finally:
                elapsed = (time.time() - start) * 1000
                if trace_id:
                    logger.info("%s took %.2f ms trace=%s", label, elapsed, trace_id)
                else:
                    logger.info("%s took %.2f ms", label, elapsed)
        return wrapper
    return decorator
