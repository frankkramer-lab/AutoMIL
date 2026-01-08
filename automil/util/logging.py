"""
Project wide logging utilities for AutoMIL.
"""

from typing import Callable

from slideflow.util import log as slideflow_log

from .enums import LogLevel


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
