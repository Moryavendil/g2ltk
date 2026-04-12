from typing import Optional, Any, Tuple, Dict, List, Union
import numpy as np
import math

from .. import set_verbose, log_error, log_warn, log_warning, log_info, log_subinfo, log_debug, log_trace, log_subtrace

### ARRAYS QOL routines
# These are the real deal, with the dimensions encoded.
# floatarray1D = np.ndarray[tuple[int], np.dtype[np.floating]]
# floatarray2D = np.ndarray[tuple[int, int], np.dtype[np.floating]]
# complexarray1D = np.ndarray[tuple[int], np.dtype[np.inexact]]
# complexarray2D = np.ndarray[tuple[int, int], np.dtype[np.inexact]]
# now using them is a pain in the ass, so we relax the dimension
floatarray1D = np.ndarray[tuple[int, ...], np.dtype[np.floating]]
floatarray2D = np.ndarray[tuple[int, ...], np.dtype[np.floating]]
complexarray1D = np.ndarray[tuple[int, ...], np.dtype[np.inexact]]
complexarray2D = np.ndarray[tuple[int, ...], np.dtype[np.inexact]]

### log
def attenuate_power(value, attenuation_factor_dB):
    return value / math.pow(10, attenuation_factor_dB / 20)


def log_amplitude_range(maximum_amplitude: float, range_db: Union[float, int]):
    return maximum_amplitude, attenuate_power(maximum_amplitude, range_db)


def log_amplitude_cbticks(maximum_amplitude: float, range_db: Union[int, float]):
    step_major = 20
    step_minor = 5
    if range_db < 60:
        step_major = 10
        step_minor = 2
    if range_db < 30:
        step_major = 5
        step_minor = 1
    # it seems unreasonable to have range_db > 100 or < 10
    att_db_major = np.arange(0, range_db + 1, step_major)
    att_db_minor = np.arange(0, range_db + 1, step_minor)
    cbticks_major = [attenuate_power(maximum_amplitude, att_db) for att_db in att_db_major]
    cbticks_minor = [attenuate_power(maximum_amplitude, att_db) for att_db in att_db_minor]
    cbticklabels = ['0 dB' if att_db == 0 else f'-{att_db} dB' for att_db in att_db_major]
    return cbticks_major, cbticklabels


### FFT AND PSD COMPUTATIONS

from .FFT1D import *

from .FFT2D import *
