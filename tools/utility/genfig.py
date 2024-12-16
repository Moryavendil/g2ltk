from typing import Optional, Any, Tuple, Dict, List, Union
import numpy as np

from .. import display, throw_G2L_warning, log_error, log_warn, log_info, log_debug, log_trace, log_subtrace

########## SAVE GRAPHE
import os
import matplotlib.pyplot as plt

in_per_mm = 1 / 25.4
screen_dpi = 122.38 # 24'' , 2560x1440 px screen. Use 91.79 for 24'' FHD and 165.63 for 13.3'' FHD  (default 100)

figw_aps:Dict[str, float] = {'simple': 86 * in_per_mm, 'wide': 140 * in_per_mm, 'double': 180 * in_per_mm,
                             'inset': 40 * in_per_mm}
figw_confort:Dict[str, float] = {'simple': 120*in_per_mm, 'wide': 190*in_per_mm, 'double': 250*in_per_mm,
                                 'inset': 60*in_per_mm}

figw = {**figw_confort}

def figsize(w:Optional[Union[float, int, str]], h:Optional[Union[float, int, str]], unit='mm', ratio = 1.618) ->Tuple[float, float]:
    global in_per_mm
    width_in = figw['simple']
    if isinstance(w, str):
        width_in = figw.get(w, None)
        if width_in is None:
            log_error('Unrecognized figsize: {w}'.format(w=w))
    elif unit == 'mm':
        width_in = float(w)*in_per_mm
    elif unit == 'in':
        width_in = float(w)
    elif w is not None:
        log_error('Unrecognized unit for figsize: {unit}'.format(unit=unit))

    height_in = width_in / ratio
    if isinstance(h, str):
        height_in = figw.get(h, None)
        if height_in is None:
            log_error('Unrecognized figsize: {h}'.format(h=h))
    elif unit == 'mm':
        height_in = float(h)*in_per_mm
    elif unit == 'in':
        height_in = float(h)
    elif h is not None:
        log_error('Unrecognized unit for figsize: {unit}'.format(unit=unit))

    return (width_in, height_in)

def configure_mpl(font_size=12):
    # figure options
    global screen_dpi
    plt.rcParams["figure.dpi"] = screen_dpi
    plt.rcParams["figure.max_open_warning"] = 50 # we have RAM

    # use confortable figure size
    global figw, figw_confort
    figw = {**figw_confort}
    figwidth = figw['double']
    figheight = figwidth / 1.618 # golden ratio
    plt.rcParams["figure.figsize"] = (figwidth, figheight)

    # Specific plots
    plt.rcParams["image.interpolation"] = "nearest" # for science, to not be misleading (default antialiased)
    plt.rcParams["errorbar.capsize"] = 3 # have small caps in errorbars (default 0)
    plt.rcParams["hist.bins"] = 20 # more bins by default (default 10)
    plt.rcParams["legend.labelspacing"] = 0.3 # less space between legends items (default 0.5)


    # have readable font size
    plt.rcParams.update({'font.family': 'serif', 'font.size': font_size,
                         'legend.fontsize': font_size,
                         'axes.labelsize': font_size, 'axes.titlesize': font_size,
                         'figure.labelsize': font_size,
                         })

    # setup correct
    plt.rcParams['pgf.texsystem'] = 'pdflatex'

def activate_saveplot(activate=True, font_size=10):
    if not activate:
        deactivate_saveplot()
        return
    # use LaTeX
    plt.rcParams['text.usetex'] = True
    plt.rcParams['pgf.texsystem'] = 'pdflatex'

    # use figure size
    global figw, figw_aps
    figw = {**figw_aps}
    figwidth = figw['simple']
    figheight = figwidth / 1.618 # golden ratio
    plt.rcParams["figure.figsize"] = (figwidth, figheight)

    # have appropriate font size
    plt.rcParams.update({'font.family': 'serif', 'font.size': font_size,
                         'legend.fontsize': font_size,
                         'axes.labelsize': font_size, 'axes.titlesize': font_size,
                         'figure.labelsize': font_size,
                         })
    # saving options
    plt.rcParams.update({'savefig.bbox': 'tight', # tight or standard
                         'savefig.dpi': 300, # default 'figure'
                         'savefig.pad_inches': 0., # padding to be used, when bbox is set to 'tight' ; default 0.1
                         'savefig.transparent': True,
                         # # tight layout
                         # 'figure.subplot.hspace': 0., 'figure.subplot.wspace': 0.,
                         # # 'figure.subplot.hspace': 0.2, 'figure.subplot.wspace': 0.2,
                         # 'figure.subplot.left': 0, 'figure.subplot.right': 1.,
                         # 'figure.subplot.top': 1., 'figure.subplot.bottom': 0.,
                         # # constrained layout
                         # 'figure.constrained_layout.h_pad': 0.,
                         # 'figure.constrained_layout.w_pad': 0.,
                         })

def deactivate_saveplot(font_size=12):
    # stop using LaTeX for faster display
    plt.rcParams['text.usetex'] = False

    configure_mpl(font_size=font_size)
def tighten_graph(pad=0., w_pad=0., h_pad=0.):
    plt.tight_layout(pad=pad, w_pad=w_pad, h_pad=h_pad)
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