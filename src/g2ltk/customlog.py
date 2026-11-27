"""
Logging for g2ltk, built on Python's standard ``logging`` library.

Every message goes through the ``'g2ltk'`` logger. By default g2ltk installs no handler, so the
standard library behaviour applies: warnings and above are printed plainly to stderr, and the rest
is dropped, unless the application configured logging itself.

Calling ``set_verbose(level)`` sets the verbosity and switches to the g2ltk display style
(coloured, prefixed and indented messages on stdout).

The levels, from least to most verbose, are:
critical, error, warning, info, subinfo, debug, trace, subtrace.
"""

import contextlib
import logging
import sys
from typing import Iterator, Union

from g2ltk import throw_G2L_warning

__all__ = ['SUBINFO', 'TRACE', 'SUBTRACE', 'LEVELS', 'logger', 'set_verbose', 'verbosity',
           'log_criticalFailure', 'log_error', 'log_warning', 'log_info', 'log_subinfo',
           'log_debug', 'log_trace', 'log_subtrace']

# Custom levels, in the standard numbering (a lower number means more verbose)
SUBINFO = 15  # between INFO (20) and DEBUG (10)
TRACE = 5
SUBTRACE = 3
for _level, _name in ((SUBINFO, 'SUBINFO'), (TRACE, 'TRACE'), (SUBTRACE, 'SUBTRACE')):
    logging.addLevelName(_level, _name)

LEVELS = {'critical': logging.CRITICAL,
          'error': logging.ERROR,
          'warning': logging.WARNING,
          'info': logging.INFO,
          'subinfo': SUBINFO,
          'debug': logging.DEBUG,
          'trace': TRACE,
          'subtrace': SUBTRACE}

default_verbose = 'info'

logger = logging.getLogger('g2ltk')


class _Fore:
    """ANSI escape codes to change the display text color (same names as colorama.Fore)"""
    GREEN = '\x1b[32m'
    LIGHTRED_EX = '\x1b[91m'
    LIGHTGREEN_EX = '\x1b[92m'
    LIGHTYELLOW_EX = '\x1b[93m'
    LIGHTBLUE_EX = '\x1b[94m'
    LIGHTMAGENTA_EX = '\x1b[95m'
    LIGHTCYAN_EX = '\x1b[96m'
    RESET = '\x1b[39m'


class G2LFormatter(logging.Formatter):
    """The g2ltk display style: a coloured, prefixed and indented message.

    The message starts with a carriage return and is padded to 80 characters, so that it
    overwrites a progress line printed with ``force_print(..., end='\\r')``.
    """
    DISPLAYSIZE = 80
    STYLES = {logging.CRITICAL: (_Fore.LIGHTMAGENTA_EX, 'CRITICAL: '),
              logging.ERROR: (_Fore.LIGHTRED_EX, 'ERROR: '),
              logging.WARNING: (_Fore.LIGHTYELLOW_EX, 'WARN: '),
              logging.INFO: (_Fore.LIGHTGREEN_EX, 'INFO: '),
              SUBINFO: (_Fore.GREEN, '(SUB)INFO: '),
              logging.DEBUG: (_Fore.LIGHTCYAN_EX, 'DEBUG:\t'),
              TRACE: (_Fore.LIGHTBLUE_EX, 'TRACE:\t\t'),
              SUBTRACE: (_Fore.LIGHTMAGENTA_EX, 'SUBTRACE:\t\t\t')}

    def format(self, record: logging.LogRecord) -> str:
        color, prefix = self.STYLES.get(record.levelno, ('', record.levelname + ': '))
        text = color + prefix + record.getMessage() + _Fore.RESET
        return '\r' + text + max(self.DISPLAYSIZE - len(text), 0) * ' '


class _StdoutHandler(logging.StreamHandler):
    """A handler writing to the *current* sys.stdout.

    It looks up sys.stdout at each message, so it follows redirections made after it was created
    (pytest's capsys, contextlib.redirect_stdout, ...).
    """
    def __init__(self):
        super().__init__()
        self.setFormatter(G2LFormatter())

    @property
    def stream(self):
        return sys.stdout

    @stream.setter
    def stream(self, value):
        pass


def _to_level(verbose: Union[int, str]) -> int:
    level = LEVELS.get(verbose.lower(), None) if isinstance(verbose, str) else verbose
    if not isinstance(level, int):
        throw_G2L_warning(f'Could not recognize verbose {verbose!r}, using "{default_verbose}".')
        level = LEVELS[default_verbose]
    return level


def set_verbose(verbose: Union[int, str] = default_verbose) -> None:
    """
    Set the g2ltk verbosity and display the logs in the g2ltk style.

    Parameters
    ----------
    verbose : str or int, optional
        a level name (``'info'``, ``'debug'``, ``'trace'``...) or a standard
        ``logging`` level number. Default is ``'info'``.
    """
    logger.setLevel(_to_level(verbose))
    if not any(isinstance(handler, _StdoutHandler) for handler in logger.handlers):
        logger.addHandler(_StdoutHandler())
        # Do not also pass the messages to the root logger: they would be printed twice
        # if the application configured logging (logging.basicConfig...)
        logger.propagate = False


@contextlib.contextmanager
def verbosity(verbose: Union[int, str]) -> Iterator[None]:
    """Temporarily set the g2ltk verbosity, for example ``with verbosity('error'): ...``."""
    previous_level = logger.level
    set_verbose(verbose)
    try:
        yield
    finally:
        logger.setLevel(previous_level)


# stacklevel=2 attributes each message to the function calling log_xxx, not to this module
def log_criticalFailure(text: str) -> None:
    logger.log(logging.CRITICAL, str(text), stacklevel=2)

def log_error(text: str) -> None:
    logger.log(logging.ERROR, str(text), stacklevel=2)

def log_warning(text: str) -> None:
    logger.log(logging.WARNING, str(text), stacklevel=2)

def log_info(text: str) -> None:
    logger.log(logging.INFO, str(text), stacklevel=2)

def log_subinfo(text: str) -> None:
    logger.log(SUBINFO, str(text), stacklevel=2)

def log_debug(text: str) -> None:
    logger.log(logging.DEBUG, str(text), stacklevel=2)

def log_trace(text: str) -> None:
    logger.log(TRACE, str(text), stacklevel=2)

def log_subtrace(text: str) -> None:
    logger.log(SUBTRACE, str(text), stacklevel=2)
