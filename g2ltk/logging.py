from typing import Union, Dict
from colorama import Fore # change display text color

from g2ltk import force_print, throw_G2L_warning

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

def log_criticalFailure(text:str, verbose:int=None): # verbose 0
    global verbose_codes, __VERBOSE__
    if verbose is None:
        verbose=__VERBOSE__
    if verbose >= verbose_codes['critical']:
        force_print('=!=!=!=!=!=!=!=!= CRITICAL FAILURES ARE NOT CODED YET =!=!=!=!=!=!=!=!=')
        force_print(Fore.LIGHTMAGENTA_EX + 'CRITICAL: ' + text + Fore.RESET)
def log_error(text:str, verbose:int=None): # verbose 1
    global verbose_codes, __VERBOSE__
    if verbose is None:
        verbose=__VERBOSE__
    if verbose >= verbose_codes['error']:
        force_print(Fore.LIGHTRED_EX + 'ERROR: ' + str(text) + Fore.RESET)
def log_warning(text:str, verbose:int=None): # verbose 2
    global verbose_codes, __VERBOSE__
    if verbose is None:
        verbose=__VERBOSE__
    if verbose >= verbose_codes['warning']:
        force_print(Fore.LIGHTYELLOW_EX + 'WARN: ' + str(text) + Fore.RESET)
def log_warn(text:str, verbose:int=None): # DEPRECATED
    log_warning(text, verbose=verbose)
def log_info(text:str, verbose:int=None): # verbose 3
    global verbose_codes, __VERBOSE__
    if verbose is None:
        verbose=__VERBOSE__
    if verbose >= verbose_codes['info']:
        force_print(Fore.LIGHTGREEN_EX + 'INFO: ' + str(text) + Fore.RESET)
def log_subinfo(text:str, verbose:int=None): # verbose 3.5
    global verbose_codes, __VERBOSE__
    if verbose is None:
        verbose=__VERBOSE__
    if verbose >= verbose_codes['subinfo']:
        force_print(Fore.GREEN + '(SUB)INFO: ' + str(text) + Fore.RESET)
def log_debug(text:str, verbose:int=None): # verbose 4
    global verbose_codes, __VERBOSE__
    if verbose is None:
        verbose=__VERBOSE__
    if verbose >= verbose_codes['debug']:
        force_print(Fore.LIGHTCYAN_EX + 'DEBUG:\t' + str(text) + Fore.RESET)
def log_trace(text:str, verbose:int=None): # verbose 5
    global verbose_codes, __VERBOSE__
    if verbose is None:
        verbose=__VERBOSE__
    if verbose >= verbose_codes['trace']:
        force_print(Fore.LIGHTBLUE_EX + 'TRACE:\t\t' + str(text) + Fore.RESET)
def log_subtrace(text:str, verbose:int=None): # verbose 6
    global verbose_codes, __VERBOSE__
    if verbose is None:
        verbose=__VERBOSE__
    if verbose >= verbose_codes['subtrace']:
        force_print(Fore.LIGHTMAGENTA_EX + 'SUBTRACE:\t\t\t' + str(text) + Fore.RESET)
