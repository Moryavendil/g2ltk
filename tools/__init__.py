from typing import Optional, Any, Tuple, Dict, List
import os
import numpy as np
from colorama import Fore # change display text color

DISPLAYSIZE:int = 80

def display(text:str, flush:bool=True, end:str='\n', padding:bool=True, displaytype:str=""):
    text = str(text)
    text += max(DISPLAYSIZE-len(text),0) * ' ' * padding
    print(text, flush=flush, end=end)

VIOLET = '#4f0e51'
JAUNE = '#eaac3f'

VERSION = '0.9.2'
display('tools version '+VERSION)

import warnings
_warn_skips = (os.path.dirname(__file__),)

class G2LWarning(UserWarning):
    pass

import sys
def throw_G2L_warning(text: str):
    warnings.warn(Fore.RED + text + Fore.LIGHTWHITE_EX, category=G2LWarning, stacklevel=3)
    # warnings.warn(Fore.RED + text + Fore.LIGHTWHITE_EX, category=G2LWarning, stacklevel=3,
    #               skip_file_prefixes=_warn_skips) # THIS ONLY WORKS FOR PYTHON >= 3.12
    sys.stderr.flush()
    display('')
