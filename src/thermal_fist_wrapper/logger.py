from __future__ import annotations

import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.logging import RichHandler
from rich.traceback import install as install_rich_traceback

_CONFIGURED = False
_CONSOLE: Optional[Console] = None


def get_console() -> Console:
    """Get (and create if needed) the global Rich console for logging."""
    global _CONSOLE
    if _CONSOLE is None:
        _CONSOLE = Console()
    return _CONSOLE


def _coerce_level(level: int | str) -> int:
    return logging.getLevelName(level.upper()) if isinstance(level, str) else int(level)


def _env_level(default: str = "INFO") -> int:
    level = os.getenv("LOG_LEVEL") or os.getenv("PYTHONLOGLEVEL") or default
    return _coerce_level(level)


def _attach_file_handler(
    root: logging.Logger, path: str, level: int, max_bytes: int, backup_count: int
) -> None:
    # Avoid duplicate file handlers to the same file
    for h in root.handlers:
        if isinstance(h, RotatingFileHandler) and getattr(
            h, "baseFilename", None
        ) == os.path.abspath(path):
            return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fh = RotatingFileHandler(path, maxBytes=max_bytes, backupCount=backup_count)
    fh.setLevel(level)
    fh.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s %(levelname)-8s %(filename)s:%(lineno)d \t %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    root.addHandler(fh)


def setup_logging(
    level: int | str | None = None,
    *,
    log_file: Optional[str] = None,
    file_level: (
        int | str | None
    ) = None,  # <- set to "INFO" if you want file logs even without -v
    max_bytes: int = 5 * 1024 * 1024,
    backup_count: int = 2,
    show_time: bool = True,
    show_path: bool = True,
    rich_tracebacks: bool = True,
    force_reconfigure: bool = False,
) -> None:
    """
    Configure Rich console logging and (optionally) a rotating file handler.
    Safe to call multiple times; use force_reconfigure=True to rebuild console handler.
    """
    global _CONFIGURED

    root = logging.getLogger()

    if not _CONFIGURED or force_reconfigure:
        if rich_tracebacks:
            install_rich_traceback(show_locals=False)

        # Reset handlers when reconfiguring
        if force_reconfigure:
            for h in list(root.handlers):
                root.removeHandler(h)

        root.setLevel(_env_level() if level is None else _coerce_level(level))

        rich_handler = RichHandler(
            console=get_console(),
            show_time=show_time,
            show_level=True,
            show_path=show_path,
            markup=True,
            rich_tracebacks=rich_tracebacks,
            tracebacks_show_locals=False,
            enable_link_path=True,
            log_time_format="%H:%M:%S",
        )
        rich_handler.setFormatter(logging.Formatter("%(message)s"))
        root.addHandler(rich_handler)
        _CONFIGURED = True

    # Attach/ensure file handler if requested (even if already configured before)
    if log_file:
        eff_file_level = (
            _coerce_level(file_level) if file_level is not None else root.level
        )
        _attach_file_handler(root, log_file, eff_file_level, max_bytes, backup_count)


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Get a logger for the given name (or __name__ if None)."""
    # No auto-config here; just hand back a logger that will propagate to the root
    return logging.getLogger(name if name else __name__)