from typing import Optional, Any, Tuple, Dict, List
import os
import numpy as np
import matplotlib.pyplot as plt
from colorama import Fore # change display text color

DISPLAYSIZE:int = 80

def display(text:str, flush:bool=True, end:str='\n', padding:bool=True, displaytype:str=""):
    text = str(text)
    text = '\r' + text + max(DISPLAYSIZE-len(text),0) * ' ' * padding
    print(text, flush=flush, end=end)

def save_graphe(graph_name, imageonly=False, **kwargs):
    figures_directory = 'figures'
    if not os.path.isdir(figures_directory):
        os.mkdir(figures_directory)
    raw_path = os.path.join(figures_directory, graph_name)
    if 'bbox_inches' not in kwargs:
        kwargs['bbox_inches'] = 'tight'
    if 'pad_inches' not in kwargs:
        kwargs['pad_inches'] = 0
    if imageonly:
        plt.savefig(raw_path + '.jpg', **kwargs)
    else:
        if 'dpi' not in kwargs:
            kwargs['dpi'] = 600
        plt.savefig(raw_path + '.png', **kwargs)
        plt.savefig(raw_path + '.pdf', **kwargs)
        plt.savefig(raw_path + '.svg', **kwargs)

VIOLET = '#4f0e51'
JAUNE = '#eaac3f'

__version__ = '0.9.5'
VERSION = __version__
display('tools version '+__version__)

import warnings
_warn_skips = (os.path.dirname(__file__),)

class G2LWarning(UserWarning):
    pass

import sys
def throw_G2L_warning(text:str):
    warnings.warn(Fore.LIGHTYELLOW_EX + 'Warning: ' + text + Fore.RESET, category=G2LWarning, stacklevel=3
                  # skip_file_prefixes=_warn_skips # THIS ONLY WORKS FOR PYTHON >= 3.12
                  )
    sys.stderr.flush() ; display('') # force to display warning at runtime



def log_criticalFailure(text:str, verbose:int): # verbose 0
    if verbose >= 0:
        display('=!=!=!=!=!=!=!=!= CRITICAL FAILURES ARE NOT CODED YET =!=!=!=!=!=!=!=!=')
        display(Fore.LIGHTMAGENTA_EX + 'CRITICAL: ' + text + Fore.RESET)

def log_error(text:str, verbose:int): # verbose 1
    if verbose >= 1:
        display('=!=!=!=!=!=!=!=!= ERRORS ARE NOT CODED YET =!=!=!=!=!=!=!=!=')
        display(Fore.LIGHTRED_EX + 'ERROR: ' + text + Fore.RESET)

def log_warn(text:str, verbose:int): # verbose 2
    if verbose >= 2:
        display(Fore.LIGHTYELLOW_EX + 'WARN: ' + text + Fore.RESET)

def log_info(text:str, verbose:int): # verbose 3
    if verbose >= 3:
        display(Fore.LIGHTGREEN_EX + 'INFO: ' + text + Fore.RESET)

def log_dbug(text:str, verbose:int): # verbose 4
    if verbose >= 4:
        display(Fore.LIGHTCYAN_EX + 'DEBUG:\t' + text + Fore.RESET)

def log_trace(text:str, verbose:int): # verbose 5
    if verbose >= 5:
        display(Fore.LIGHTBLUE_EX + 'TRACE:\t\t' + text + Fore.RESET)
