import logging
from logging.handlers import RotatingFileHandler
import os


def _ensure_dir() -> None:
    os.makedirs("logs", exist_ok=True)


def get_logger(name: str, filename: str = "training.log") -> logging.Logger:
    _ensure_dir()
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s")

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)

    fh = RotatingFileHandler(
        os.path.join("logs", filename), maxBytes=1_000_000, backupCount=3
    )
    fh.setFormatter(fmt)

    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger
