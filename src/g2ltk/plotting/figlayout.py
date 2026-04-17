from typing import Union, Optional, Tuple
import matplotlib.pyplot as plt

from . import in_per_mm
from g2ltk.plotting import in_per_mm
from g2ltk import log_error, log_warn, log_info, log_debug, log_trace, log_subtrace


def figsize(w:Optional[Union[float, int, str]], h:Optional[Union[float, int, str]]=None,
            ratio:Optional[float]=None, unit: str='mm') ->Tuple[float, float]:
    default_width = 'simple'
    width_in = figw[default_width]
    if isinstance(w, str):
        width_in = figw.get(w, None)
        if width_in is None:
            log_error(f'Unrecognized figsize: {w}. Using {default_width}.')
            width_in = figw[default_width]
    elif unit == 'mm':
        width_in = float(w)*in_per_mm
    elif unit == 'in':
        width_in = float(w)
    elif w is not None:
        log_error('Unrecognized unit for figsize: {unit}'.format(unit=unit))

    height_in = width_in / 1.618
    if h is None:
        if ratio is not None:
            try:
                ratio = float(ratio)
            except:
                log_error('This is not a ratio: {ratio}. Taking None instead.'.format(ratio=ratio))
        if ratio is None:
            ratio = 1.618
        height_in = width_in / ratio # by default
    else:
        if isinstance(h, str):
            height_in = figw.get(h, None)
            if height_in is None:
                log_error('Unrecognized figsize: {h}'.format(h=h))
        elif unit == 'mm':
            height_in = float(h)*in_per_mm
        elif unit == 'in':
            height_in = float(h)
        else:
            log_error('Unrecognized unit for figsize: {unit}'.format(unit=unit))

    return (width_in, height_in)

def force_aspect_ratio(ax: plt.Axes, aspect=1):
    # old version, for images
    # im = ax.get_images()
    # extent =  im[0].get_extent()
    extent = [*ax.get_xlim(), *ax.get_ylim()]
    ax.set_aspect(abs((extent[1]-extent[0])/(extent[3]-extent[2]))/aspect)

    # # IF THIS DOES NOT WORK, TRY
    # ax.set_box_aspect(1.)


def subplots_adjust(fig: plt.Figure, left=None, bottom=None, right=None, top=None, wspace=None, hspace=None, unit='rel'):
    width_in, height_in = fig.get_size_inches()

    factor_w, factor_h = 1, 1
    if unit == 'mm': # millimeters
        factor_w = in_per_mm
        factor_h = in_per_mm
    elif unit == 'in': # inches
        factor_w = 1
        factor_h = 1
    elif unit == 'rel': # relative (the usual way, but now can be negative)
        factor_w = width_in
        factor_h = height_in
    else:
        log_error('Unrecognized unit: {unit}'.format(unit=unit))
    log_subtrace(f'subplots_adjust: unit is {unit}, factor is {factor_w, factor_h}')


    if left is not None:
        left_in = left * factor_w
        if left_in >= 0:
            left = left_in / width_in
        else:
            left = (width_in - (-left_in)) / width_in

    if right is not None:
        right_in = right * factor_w
        if right_in >= 0:
            right = right_in / width_in
        else:
            right = (width_in - (-right_in)) / width_in

    if bottom is not None:
        bottom_in = bottom * factor_h
        if bottom_in >= 0:
            bottom = bottom_in / height_in
        else:
            bottom = (height_in - (-bottom_in)) / height_in

    if top is not None:
        top_in = top * factor_h
        if top_in >= 0:
            top = top_in / height_in
        else:
            top = (height_in - (-top_in)) / height_in

    if wspace is not None:
        wspace_in = wspace * factor_w
        wspace = wspace_in / width_in

    if hspace is not None:
        hspace_in = hspace * factor_h
        hspace = hspace_in / width_in

    log_trace('lauching subplots_adjust with')
    log_trace(f'    left={left} | bottom={bottom} | right={right} | top={top}')
    log_trace(f'    wspace={wspace} | hspace={hspace}')
    fig.subplots_adjust(left, bottom, right, top, wspace=wspace, hspace=hspace)

