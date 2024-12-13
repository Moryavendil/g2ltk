import os
import sys
from colorama import Fore # change display text color
from typing import Union

DISPLAYSIZE:int = 80

def display(text:str, flush:bool=True, end:str='\n', padding:bool=True, displaytype:str=""):
    text = str(text)
    text = '\r' + text + max(DISPLAYSIZE-len(text),0) * ' ' * padding
    print(text, flush=flush, end=end)

VIOLET = '#4f0e51'
JAUNE = '#eaac3f'

__version__ = '0.10.3.dev6'
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

global_verbose = 3

def set_verbose(verbose:Union[int, str]):
    global global_verbose
    if isinstance(verbose, str):
        if verbose == 'critical':
            verbose = 0
        elif verbose == 'error':
            verbose = 1
        elif verbose == 'warning':
            verbose = 2
        elif verbose == 'info':
            verbose = 3
        elif verbose == 'debug':
            verbose = 4
        elif verbose == 'trace':
            verbose = 5
        elif verbose == 'subtrace':
            verbose = 6
        else:
            throw_G2L_warning(f'Could not recognize verbose {verbose}, using 3 (info).')
            verbose = 3
    global_verbose = verbose

def log_criticalFailure(text:str, verbose:int=None): # verbose 0
    if verbose is None:
        global global_verbose
        verbose=global_verbose
    if verbose >= 0:
        display('=!=!=!=!=!=!=!=!= CRITICAL FAILURES ARE NOT CODED YET =!=!=!=!=!=!=!=!=')
        display(Fore.LIGHTMAGENTA_EX + 'CRITICAL: ' + text + Fore.RESET)

def log_error(text:str, verbose:int=None): # verbose 1
    if verbose is None:
        global global_verbose
        verbose=global_verbose
    if verbose >= 1:
        # display('=!=!=!=!=!=!=!=!= ERRORS ARE NOT CODED YET =!=!=!=!=!=!=!=!=')
        display(Fore.LIGHTRED_EX + 'ERROR: ' + str(text) + Fore.RESET)

def log_warn(text:str, verbose:int=None): # verbose 2
    if verbose is None:
        global global_verbose
        verbose=global_verbose
    if verbose >= 2:
        display(Fore.LIGHTYELLOW_EX + 'WARN: ' + str(text) + Fore.RESET)

def log_info(text:str, verbose:int=None): # verbose 3
    if verbose is None:
        global global_verbose
        verbose=global_verbose
    if verbose >= 3:
        display(Fore.LIGHTGREEN_EX + 'INFO: ' + str(text) + Fore.RESET)

def log_debug(text:str, verbose:int=None): # verbose 4
    if verbose is None:
        global global_verbose
        verbose=global_verbose
    if verbose >= 4:
        display(Fore.LIGHTCYAN_EX + 'DEBUG:\t' + str(text) + Fore.RESET)

def log_trace(text:str, verbose:int=None): # verbose 5
    if verbose is None:
        global global_verbose
        verbose=global_verbose
    if verbose >= 5:
        display(Fore.LIGHTBLUE_EX + 'TRACE:\t\t' + str(text) + Fore.RESET)

def log_subtrace(text:str, verbose:int=None): # verbose 6
    if verbose is None:
        global global_verbose
        verbose=global_verbose
    if verbose >= 6:
        display(Fore.LIGHTMAGENTA_EX + 'RETRACE:\t\t\t' + str(text) + Fore.RESET)
        

log_info('Loading tools version '+__version__)
