import os
import sys
from colorama import Fore # change display text color
from typing import Union, Dict

DISPLAYSIZE:int = 80

def display(text:str, flush:bool=True, end:str='\n', padding:bool=True, displaytype:str=""):
    text = str(text)
    text = '\r' + text + max(DISPLAYSIZE-len(text),0) * ' ' * padding
    print(text, flush=flush, end=end)

VIOLET = '#4f0e51'
JAUNE = '#eaac3f'

__version__ = '1.1.1'
VERSION = __version__

import warnings
_warn_skips = (os.path.dirname(__file__),)

class G2LWarning(UserWarning):
    pass

def throw_G2L_warning(text:str):
    warnings.warn(Fore.LIGHTYELLOW_EX + 'Warning: ' + text + Fore.RESET, category=G2LWarning, stacklevel=3
                  # skip_file_prefixes=_warn_skips # THIS ONLY WORKS FOR PYTHON >= 3.12
                  )
    sys.stderr.flush() ; display('') # force to display warning at runtime


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
        display('=!=!=!=!=!=!=!=!= CRITICAL FAILURES ARE NOT CODED YET =!=!=!=!=!=!=!=!=')
        display(Fore.LIGHTMAGENTA_EX + 'CRITICAL: ' + text + Fore.RESET)

def log_error(text:str, verbose:int=None): # verbose 1
    global verbose_codes, __VERBOSE__
    if verbose is None:
        verbose=__VERBOSE__
    if verbose >= verbose_codes['error']:
        display(Fore.LIGHTRED_EX + 'ERROR: ' + str(text) + Fore.RESET)

def log_warning(text:str, verbose:int=None): # verbose 2
    global verbose_codes, __VERBOSE__
    if verbose is None:
        verbose=__VERBOSE__
    if verbose >= verbose_codes['warning']:
        display(Fore.LIGHTYELLOW_EX + 'WARN: ' + str(text) + Fore.RESET)

def log_warn(text:str, verbose:int=None): # DEPRECATED
    log_warning(text, verbose=verbose)
def log_info(text:str, verbose:int=None): # verbose 3
    global verbose_codes, __VERBOSE__
    if verbose is None:
        verbose=__VERBOSE__
    if verbose >= verbose_codes['info']:
        display(Fore.LIGHTGREEN_EX + 'INFO: ' + str(text) + Fore.RESET)
def log_subinfo(text:str, verbose:int=None): # verbose 3.5
    global verbose_codes, __VERBOSE__
    if verbose is None:
        verbose=__VERBOSE__
    if verbose >= verbose_codes['subinfo']:
        display(Fore.GREEN + '(SUB)INFO: ' + str(text) + Fore.RESET)
def log_debug(text:str, verbose:int=None): # verbose 4
    global verbose_codes, __VERBOSE__
    if verbose is None:
        verbose=__VERBOSE__
    if verbose >= verbose_codes['debug']:
        display(Fore.LIGHTCYAN_EX + 'DEBUG:\t' + str(text) + Fore.RESET)

def log_trace(text:str, verbose:int=None): # verbose 5
    global verbose_codes, __VERBOSE__
    if verbose is None:
        verbose=__VERBOSE__
    if verbose >= verbose_codes['trace']:
        display(Fore.LIGHTBLUE_EX + 'TRACE:\t\t' + str(text) + Fore.RESET)

def log_subtrace(text:str, verbose:int=None): # verbose 6
    global verbose_codes, __VERBOSE__
    if verbose is None:
        verbose=__VERBOSE__
    if verbose >= verbose_codes['subtrace']:
        display(Fore.LIGHTMAGENTA_EX + 'SUBTRACE:\t\t\t' + str(text) + Fore.RESET)
        

log_info('Loading g2ltk version '+__version__)
