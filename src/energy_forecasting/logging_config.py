"""Logging setup for the package.

Call ``setup_logging()`` once at the start of a script/app, then use
``get_logger(__name__)`` in modules. If ``configs/logging.yaml`` is present it is
used; otherwise a sensible console default is applied.
"""

from __future__ import annotations

import logging
import logging.config

from .settings import CONFIG_DIR, get_settings

_CONFIGURED = False
_DEFAULT_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"


def setup_logging(level: str | None = None) -> None:
    """Configure logging from ``configs/logging.yaml`` or a console default.

    Idempotent: safe to call multiple times. ``level`` overrides the root level.
    """
    global _CONFIGURED

    log_yaml = CONFIG_DIR / "logging.yaml"
    if log_yaml.exists():
        import yaml

        with log_yaml.open("r", encoding="utf-8") as fh:
            logging.config.dictConfig(yaml.safe_load(fh))
    else:
        logging.basicConfig(
            level=level or get_settings().log_level,
            format=_DEFAULT_FORMAT,
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    if level:
        logging.getLogger().setLevel(level)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a logger, configuring logging on first use."""
    if not _CONFIGURED:
        setup_logging()
    return logging.getLogger(name)
