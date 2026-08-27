"""Logging setup: readable console output, full detail in a rotating file."""

from __future__ import annotations

import logging
import logging.handlers
from pathlib import Path

FORMAT = "%(asctime)s %(levelname)-8s %(name)-38s %(message)s"


def configure_logging(level: str = "INFO", log_dir: Path | None = None) -> None:
    """Configure the root logger once, for the whole process."""
    root = logging.getLogger()
    if root.handlers:
        return

    root.setLevel(logging.DEBUG)
    formatter = logging.Formatter(FORMAT, datefmt="%H:%M:%S")

    console = logging.StreamHandler()
    console.setLevel(getattr(logging, level.upper(), logging.INFO))
    console.setFormatter(formatter)
    root.addHandler(console)

    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        # Stack traces stay here rather than going to the browser.
        file_handler = logging.handlers.RotatingFileHandler(
            log_dir / "backend.log", maxBytes=5_000_000, backupCount=3, encoding="utf-8",
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    for noisy in ("httpx", "httpcore", "PIL", "urllib3", "matplotlib"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
