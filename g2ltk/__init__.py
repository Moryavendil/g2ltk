import os
import sys
from colorama import Fore # change display text color

__DISPLAYSIZE__:int = 80

def force_print(text:str, flush:bool=True, end:str='\n', padding:bool=True, displaytype:str=""):
    text = str(text)
    text = '\r' + text + max(__DISPLAYSIZE__ - len(text), 0) * ' ' * padding
    print(text, flush=flush, end=end)


VIOLET = '#4f0e51'
JAUNE = '#eaac3f'

__version__ = '1.1.7'

import warnings
_warn_skips = (os.path.dirname(__file__),)

class G2LWarning(UserWarning):
    pass

def throw_G2L_warning(text:str):
    warnings.warn(Fore.LIGHTYELLOW_EX + 'Warning: ' + text + Fore.RESET, category=G2LWarning, stacklevel=3
                  # skip_file_prefixes=_warn_skips # THIS ONLY WORKS FOR PYTHON >= 3.12
                  )
    sys.stderr.flush() ; force_print('') # force to display warning at runtime

from .logging import *

log_info('Loading g2ltk version '+__version__)


def display(text:str, flush:bool=True, end:str='\n', padding:bool=True, displaytype:str=""):
    log_warning('The use of g2ltk.display function is deprecated !')
    force_print(text, flush=flush, end=end, padding=padding, displaytype=displaytype)
        

