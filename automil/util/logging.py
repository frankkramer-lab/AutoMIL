"""
Project wide logging utilities for AutoMIL.
"""

from io import StringIO
from typing import Callable

from rich.box import HEAVY_HEAD
from rich.console import Console
from rich.table import Table
from slideflow.util import log as slideflow_log

from .enums import LogLevel


# --- Logger --- #
def get_vlog(verbose: bool) -> Callable:
    def _vlog(message: str, level: LogLevel = LogLevel.INFO):
        if not verbose:
            return
        match level:
            case LogLevel.INFO:
                slideflow_log.info(message)
            case LogLevel.DEBUG:
                slideflow_log.debug(message)
            case LogLevel.WARNING:
                slideflow_log.warning(message)
            case LogLevel.ERROR:
                slideflow_log.error(message)
    return _vlog

# --- Tables ---#
def render_kv_table(
    rows: list[tuple[str, str]],
    *,
    title: str | None = None,
    width: int = 160,
) -> str:
    table = Table(
        title=title,
        show_header=False,
        show_edge=True,
        pad_edge=False,
        box=HEAVY_HEAD,
    )

    table.add_column(
        justify="left",
        no_wrap=True,
        style="bold",
    )
    table.add_column(
        justify="left",
        no_wrap=True,   # IMPORTANT: prevent wrapping
        overflow="ellipsis",  # truncate long paths cleanly
    )

    for key, value in rows:
        table.add_row(str(key), str(value))

    buffer = StringIO()
    console = Console(
        file=buffer,
        force_terminal=True,
        color_system=None,
        width=width,
    )
    console.print(table)

    return buffer.getvalue()
