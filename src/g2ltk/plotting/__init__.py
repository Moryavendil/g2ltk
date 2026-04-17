import os
import matplotlib.pyplot as plt

from g2ltk.logging import log_debug

# conversion to in
in_per_mm = 1 / 25.4
in_per_pt = 0.01384
in_per_pc = 0.16605

from styling import configure_mpl, set_figw, latex_preamble

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

def is_saveplot_activated():
    return plt.rcParams['text.usetex']


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

from colorscheme import *

from figlayout import *

from axessetup import *