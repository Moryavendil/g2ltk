from typing import Union, Dict

from g2ltk import force_print, throw_G2L_warning


class Fore:
    """ANSI escape codes to change the display text color (same names as colorama.Fore)"""
    GREEN = '\x1b[32m'
    LIGHTRED_EX = '\x1b[91m'
    LIGHTGREEN_EX = '\x1b[92m'
    LIGHTYELLOW_EX = '\x1b[93m'
    LIGHTBLUE_EX = '\x1b[94m'
    LIGHTMAGENTA_EX = '\x1b[95m'
    LIGHTCYAN_EX = '\x1b[96m'
    RESET = '\x1b[39m'

"""
This absolutely horrendous portion of code is not meant to be read or used by anyone other than me.

Everything here should be scraped and rebuilt using Python's standard customlog library.

I am just too lazy to do that for now.
"""

verbose_codes:Dict[str, int] = {'critical': 0,
                                'error': 10,
                                'warning': 20,
                                'info': 30,
                                'subinfo': 35,
                                'debug': 40,
                                'trace': 50,
                                'subtrace': 55,
                                'subsubtrace': 60,}

default_verbose = "info"
__VERBOSE__ = verbose_codes[default_verbose]

def set_verbose(verbose:Union[int, str]):
    global __VERBOSE__, verbose_codes
    if isinstance(verbose, str):
        verbose = verbose_codes.get(verbose, None)
    if not isinstance(verbose, int):
        global default_verbose
        throw_G2L_warning(f'Could not recognize verbose {verbose}, using "{default_verbose}".')
        verbose = verbose_codes[default_verbose]
    __VERBOSE__ = verbose

def log_criticalFailure(text:str): # verbose 0
    if __VERBOSE__ >= verbose_codes['critical']:
        force_print('=!=!=!=!=!=!=!=!= CRITICAL FAILURES ARE NOT CODED YET =!=!=!=!=!=!=!=!=')
        force_print(Fore.LIGHTMAGENTA_EX + 'CRITICAL: ' + text + Fore.RESET)
def log_error(text:str): # verbose 1
    if __VERBOSE__ >= verbose_codes['error']:
        force_print(Fore.LIGHTRED_EX + 'ERROR: ' + str(text) + Fore.RESET)
def log_warning(text:str): # verbose 2
    if __VERBOSE__ >= verbose_codes['warning']:
        force_print(Fore.LIGHTYELLOW_EX + 'WARN: ' + str(text) + Fore.RESET)
def log_info(text:str): # verbose 3
    if __VERBOSE__ >= verbose_codes['info']:
        force_print(Fore.LIGHTGREEN_EX + 'INFO: ' + str(text) + Fore.RESET)
def log_subinfo(text:str): # verbose 3.5
    if __VERBOSE__ >= verbose_codes['subinfo']:
        force_print(Fore.GREEN + '(SUB)INFO: ' + str(text) + Fore.RESET)
def log_debug(text:str): # verbose 4
    if __VERBOSE__ >= verbose_codes['debug']:
        force_print(Fore.LIGHTCYAN_EX + 'DEBUG:\t' + str(text) + Fore.RESET)
def log_trace(text:str): # verbose 5
    if __VERBOSE__ >= verbose_codes['trace']:
        force_print(Fore.LIGHTBLUE_EX + 'TRACE:\t\t' + str(text) + Fore.RESET)
def log_subtrace(text:str): # verbose 6
    if __VERBOSE__ >= verbose_codes['subtrace']:
        force_print(Fore.LIGHTMAGENTA_EX + 'SUBTRACE:\t\t\t' + str(text) + Fore.RESET)
