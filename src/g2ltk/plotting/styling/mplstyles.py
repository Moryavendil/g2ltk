### mpl params

import matplotlib.pyplot as plt

from g2ltk.logging import log_debug
from . import styled, get_screen_DPI, set_figw

# Default visible at : https://matplotlib.org/stable/users/explain/customizing.html (at leas version 3.10)


# lines, scatters and plots
params_lines = {
    'lines.linewidth': 1.5, # default 1.5
    'lines.markerfacecolor': 'w', # default 'auto'
    'lines.markersize': 5, # default 6          # marker size, in points
    "hist.bins": 100, # more bins by default (default 10)
    "errorbar.capsize": 3 # have small caps in errorbars (default 0)
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
                 # ### constrained layout
                 # # Padding (in inches) around axes; defaults to 3/72 inches, i.e. 3 points.
                 'figure.constrained_layout.h_pad':  0.0,
                 'figure.constrained_layout.w_pad':  0.0,
                 # # Spacing between subplots, relative to the subplot sizes.  Much smaller than for
                 # # tight_layout (figure.subplot.hspace, figure.subplot.wspace) as constrained_layout
                 # # already takes surrounding texts (titles, labels, # ticklabels) into account.
                 #figure.constrained_layout.hspace: 0.02
                 #figure.constrained_layout.wspace: 0.02
                 }

def params_fontsize(font_size):
    return { 'font.family': 'serif', 'font.size': font_size,
             'legend.fontsize': font_size,
             'axes.labelsize': font_size, 'axes.titlesize': font_size,
             'figure.labelsize': font_size,
             }


def apply_mpl_params(font_size):
    plt.rcParams.update(params_fontsize(font_size))
    plt.rcParams.update(params_lines)
    plt.rcParams.update(params_legend)
    plt.rcParams.update(params_saving)


def configure_mpl(font_size=None, style=None):
    # figure options
    plt.rcParams["figure.dpi"] = get_screen_DPI()
    plt.rcParams["figure.max_open_warning"] = 50 # we have RAM

    # use confortable figure size
    figw = styled('figw', style=style)
    set_figw(figw)
    figwidth = figw['double']
    figheight = figwidth / 1.618 # golden ratio
    plt.rcParams["figure.figsize"] = (figwidth, figheight)

    # use constrained_layout
    plt.rcParams['figure.constrained_layout.use'] = True

    # Specific plots
    plt.rcParams["image.interpolation"] = "nearest" # for science, to not be misleading (default antialiased)

    try:
        if font_size == 'inset':
            font_size = styled('fontsize_inset', style=style)
    except:pass
    if font_size is None:
        font_size = styled('fontsize', style=style)
    log_debug(f'Setting font size to {font_size} pt')
    apply_mpl_params(font_size)

    # setup correct
    plt.rcParams['pgf.texsystem'] = 'pdflatex'