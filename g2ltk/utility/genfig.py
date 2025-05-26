from typing import Optional, Any, Tuple, Dict, List, Union
import numpy as np

from .. import log_error, log_warn, log_info, log_debug, log_trace, log_subtrace

########## SAVE GRAPHE
import os
import matplotlib.pyplot as plt

# conversion to in
in_per_mm = 1 / 25.4
in_per_pt = 0.01384
in_per_pc = 0.16605
# screen_dpi = 100 # default 100
screen_dpi = 122.38 # 24'' 2560x1440 px screen
# screen_dpi = 91.79 # 24'' FHD
# screen_dpi = 165.63 # 13.3'' FHD

""" FIGURE STYLES
JFM - Conform with the Journal of Fluid Mechanics template
* textsize: 32 pc (=134.96 mm) | single column
* textheight: 49 baselineskip (12 pt) = 210 mm
* text font size: 10.5 pt | Fig/legend font size: 9 pt
* font: newtxt, MUST INCLUDE jfm_latex_preamble

PREZ - PowerPoint standard presentation
* half-screen : 150 mm x 150 mm squere domain for plots

THEZ - These de doctorat de G2L, template UPCité modifié par A. Briole et G2L
* largeur : 210 mm (aA4) - 30 mm (left) - 25 mm (right) = 155 mm
* font: ???, 11pt (figure: 10pt)
"""


# for APS & AIP (latex revtex4 2024)
figw_aps:Dict[str, float] = {'simple': 86 * in_per_mm, 'wide': 140 * in_per_mm, 'double': 180 * in_per_mm,
                             'inset': 40 * in_per_mm}
# for JFM (2025)
figw_jfm:Dict[str, float] = {'simple': 15*in_per_pc, 'wide': 21*in_per_pc, 'double': 32*in_per_pc,
                             'inset': 7*in_per_pc}
# for thesis - WORK IN PROGRESS
figw_thesis:Dict[str, float] = {'simple': 70 * in_per_mm, 'wide': 110 * in_per_mm, 'double': 155 * in_per_mm,
                               'inset': 35*in_per_mm, 'small': 50*in_per_mm}
# for viewing confort, on a screen
figw_confort:Dict[str, float] = {'simple': 120*in_per_mm, 'wide': 190*in_per_mm, 'double': 250*in_per_mm,
                                 'inset': 60*in_per_mm}

figw = {**figw_confort}

def figsize(w:Optional[Union[float, int, str]], h:Optional[Union[float, int, str]]=None, ratio:Optional[float]=None, unit='mm') ->Tuple[float, float]:
    global in_per_mm
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

def subplots_adjust(fig:plt.Figure, left=None, bottom=None, right=None, top=None, wspace=None, hspace=None, unit='rel'):
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

jfm_latex_preamble = r"""%
%
%%% JFM
%
\usepackage{newtxtext}
\usepackage{newtxmath}
%"""

thesis_latex_preamble = r"""%
%
%%% THESE
%
\usepackage[utf8]{inputenc}		% LaTeX, comprend les accents !
\usepackage[T1]{fontenc}
\usepackage[french,english]{babel}
\usepackage{csquotes}
\usepackage{lmodern}
\usepackage{ae,aecompl}				
\usepackage[upright]{fourier}
%"""


styledict_default = {'name': 'default', 'textfontsize': 12, 'fontsize': 10, 'latex_preamble': '%', 'figw': figw_confort}
styledict_presentation = {'name': 'presentation', 'textfontsize': 20, 'fontsize': 18}
styledict_thesis = {'name': 'thesis', 'textfontsize': 11, 'fontsize': 10, 'latex_preamble': thesis_latex_preamble, 'figw': figw_thesis}
styledict_jfm = {'name': 'jfm', 'textfontsize': 10.5, 'fontsize': 9, 'latex_preamble': jfm_latex_preamble, 'figw': figw_jfm}
styledict_aps = {'name': 'aps', 'figw': figw_aps}

styledicts = {'jfm': styledict_jfm, 'aps': styledict_aps,
              'presentation': styledict_presentation, 'thesis': styledict_thesis, 
              'default': styledict_default}

def fetch_styledict(style:Optional[str]=None):
    if style is None:
        styledict = styledict_default
    else:
        styledict = styledicts.get(style, styledict_default)
    return styledict

def styled(key:str, style:Optional[str]=None):
    styledict = fetch_styledict(style)

    return styledict.get(key, styledict_default.get(key, None))

### latex

general_latex_preamble = r"""%
%%% PACKAGES
%
\usepackage{amsmath} %maths
\usepackage{amssymb} %maths
\usepackage{amsfonts} %maths
\usepackage{xcolor} % use color

%% Physique
\usepackage{siunitx} % dimensioned quantities
\sisetup{per-mode = symbol,
	range-phrase = --,
	exponent-product = \cdot, % pour avoir 1.5e10 -> 1.5 \cdot 10^{15}
%    print-unity-mantissa=true,
%    product_units = repeat,
	inter-unit-product=\ensuremath{{}\cdot{}}, % pour avoir N \cdot m instead of N m
	separate-uncertainty=true, % pour avoir 25.8 \pm 0.2 plutôt que 25.8(2)
	range-units = single}
\DeclareSIUnit{\pixel}{px}
\DeclareSIUnit{\px}{px}
\DeclareSIUnit{\frame}{frame}
\DeclareSIUnit\litre{l} % redefine litre to have a small l

\usepackage{physics} % Physics-friendly commands

%%% QOL
%
\setlength{\parindent}{0pt}% no indent
%"""

paper_specific_latex_preamble = r"""%
%
%%% MACROS
%
\newcommand{\vdrift}{v_\text{drift}}
%
\newcommand{\mucl}{\ensuremath{\mu_\text{cl}}}
%
\newcommand{\lcap}{\ensuremath{\ell_\text{c}}} % capillary length l_c or l_cap or l_gamma
\newcommand{\vcap}{\ensuremath{v_\text{c}}} % capillary speed v_c or v_cap or v_gamma
\newcommand{\us}{{\ensuremath{u_s}}}
\newcommand{\un}{{\ensuremath{u_n}}}
%"""

def latex_preamble(style=None):
    style_specific_latex_preamble = styled('latex_preamble', style=style)

    return fr"""%
%
{general_latex_preamble}
%
{paper_specific_latex_preamble}
%
{style_specific_latex_preamble}
%
%"""

### mpl params

# lines, scatters and plots
params_lines = {
    'lines.linewidth': 1.0, # default 1.5
    'lines.markerfacecolor': 'w', # default 'auto'
    'lines.markersize': 5 # default 6          # marker size, in points
    }
# legends and axes labels
params_legend = {
    'legend.frameon': True, # default True
    'legend.framealpha': 0.9, # default 0.9
    'legend.edgecolor': '1.0', # default 0.8
    'legend.labelspacing': 0.3, # default 0.5
    'legend.handletextpad': 0.5,  # default 0.8
    'axes.titlepad': 4.0, # default 6.0
    'axes.labelpad': 1.0 # default 4.0
    }
# figures saving
params_saving = {'savefig.bbox': 'tight', # tight or standard
                     'savefig.dpi': 600, # default 'figure'
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
                     }

def params_fontsize(font_size):
    return { 'font.family': 'serif', 'font.size': font_size,
             'legend.fontsize': font_size,
             'axes.labelsize': font_size, 'axes.titlesize': font_size,
             'figure.labelsize': font_size,
             }

# Default visible at : https://matplotlib.org/stable/users/explain/customizing.html (at leas version 3.10)


def configure_mpl(font_size=None, style=None):
    # figure options
    global screen_dpi
    plt.rcParams["figure.dpi"] = screen_dpi
    plt.rcParams["figure.max_open_warning"] = 50 # we have RAM

    # use confortable figure size
    global figw
    figw = {**styled('figw', style=style)}
    figwidth = figw['double']
    figheight = figwidth / 1.618 # golden ratio
    plt.rcParams["figure.figsize"] = (figwidth, figheight)

    # Specific plots
    plt.rcParams["image.interpolation"] = "nearest" # for science, to not be misleading (default antialiased)
    plt.rcParams["errorbar.capsize"] = 3 # have small caps in errorbars (default 0)
    plt.rcParams["hist.bins"] = 20 # more bins by default (default 10)
    plt.rcParams["legend.labelspacing"] = 0.3 # less space between legends items (default 0.5)

    if font_size is None:
        font_size = styled('fontsize', style=style)
    log_debug(f'Setting font size to {font_size} pt')
    plt.rcParams.update(params_fontsize(font_size))

    global params_lines, params_legend, params_saving
    plt.rcParams.update(params_lines)
    plt.rcParams.update(params_legend)
    plt.rcParams.update(params_saving)

    # setup correct
    plt.rcParams['pgf.texsystem'] = 'pdflatex'

def activate_saveplot(activate=True, font_size=None, style=None):
    # use LaTeX
    plt.rcParams['text.usetex'] = activate
    if activate:
        plt.rcParams['pgf.texsystem'] = 'pdflatex'
        plt.rcParams['text.latex.preamble'] = latex_preamble(style=style)

    configure_mpl(font_size=font_size, style=style)

def deactivate_saveplot(font_size=None, style=None):
    # stop using LaTeX for faster display
    activate_saveplot(activate=False, font_size=font_size, style=style)
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
