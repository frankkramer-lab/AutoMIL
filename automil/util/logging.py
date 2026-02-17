#==============================================================================#
#  AutoMIL - Automated Machine Learning for Image Classification in            #
#  Whole-Slide Imaging with Multiple Instance Learning                         #
#                                                                              #
#  Copyright (C) 2026 Jonas Waibel                                             #
#                                                                              #
#  This program is free software: you can redistribute it and/or modify        #
#  it under the terms of the GNU General Public License as published by        #
#  the Free Software Foundation, either version 3 of the License, or           #
#  (at your option) any later version.                                         #
#                                                                              #
#  This program is distributed in the hope that it will be useful,             #
#  but WITHOUT ANY WARRANTY; without even the implied warranty of              #
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the               #
#  GNU General Public License for more details.                                #
#                                                                              #
#  You should have received a copy of the GNU General Public License           #
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.      #
#==============================================================================#
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
