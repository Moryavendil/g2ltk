from typing import Dict, Optional
import matplotlib.pyplot as plt
from g2ltk.plotting import in_per_mm, in_per_pc


def fetch_styledict(style:Optional[str]=None):
    if style is None:
        styledict = styledict_default
    else:
        styledict = styledicts.get(style, styledict_default)
    return styledict

def styled(key:str, style:Optional[str]=None):
    styledict = fetch_styledict(style)

    return styledict.get(key, styledict_default.get(key, None))

from latexstyles import *

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
* font: Erewhon, 11pt (figure: 10pt)
"""


# for APS & AIP (latex revtex4 2024)
figw_aps:Dict[str, float] = {'simple': 86 * in_per_mm, 'wide': 140 * in_per_mm, 'double': 180 * in_per_mm,
                             'inset': 40 * in_per_mm}
# for JFM (2025)
figw_jfm:Dict[str, float] = {'simple': 15*in_per_pc, 'wide': 21*in_per_pc, 'double': 32*in_per_pc,
                             'inset': 7*in_per_pc}
# for thesis
figw_thesis:Dict[str, float] = {'simple': 70 * in_per_mm, 'wide': 110 * in_per_mm, 'double': 155 * in_per_mm,
                                'inset': 35*in_per_mm, 'small': 50*in_per_mm}
# for thesis presentation powerpoint
figw_thezprez:Dict[str, float] = {'simple': 160 * in_per_mm, 'wide': 200 * in_per_mm, 'double': 300 * in_per_mm,
                                  'inset': 70*in_per_mm, 'small': 100*in_per_mm}
# for viewing confort, on a screen
figw_confort:Dict[str, float] = {'simple': 120*in_per_mm, 'wide': 190*in_per_mm, 'double': 250*in_per_mm,
                                 'inset': 60*in_per_mm}

figw = {}
def set_figw(new_figw:Dict[str, float]):
    global figw
    figw = {**new_figw}

set_figw(figw_confort)

styledict_default = {'name': 'default', 'textfontsize': 12, 'fontsize': 10, 'latex_preamble': '%', 'figw': figw_confort}
styledict_presentation = {'name': 'presentation', 'textfontsize': 20, 'fontsize': 18}
styledict_thesis = {'name': 'thesis', 'textfontsize': 11, 'fontsize': 10, 'fontsize_inset': 9, 'latex_preamble': thesis_latex_preamble, 'figw': figw_thesis}
styledict_thezprez = {'name': 'thezprez', 'textfontsize': 18, 'fontsize': 18, 'fontsize_inset': 16, 'latex_preamble': thesis_latex_preamble, 'figw': figw_thezprez}
styledict_jfm = {'name': 'jfm', 'textfontsize': 10.5, 'fontsize': 9, 'latex_preamble': jfm_latex_preamble, 'figw': figw_jfm}
styledict_aps = {'name': 'aps', 'textfontsize': 10, 'fontsize': 9, 'fontsize_inset': 8, 'figw': figw_aps}

styledicts = {'jfm': styledict_jfm, 'aps': styledict_aps,
              'presentation': styledict_presentation,
              'thesis': styledict_thesis, 'thezprez': styledict_thezprez,
              'default': styledict_default}





def get_screen_DPI():
    # # To get DPI:
    # from PyQt5.QtWidgets import QApplication
    # import sys
    # app = QApplication(sys.argv)
    # screen = app.primaryScreen()
    # print("DPI:", screen.logicalDotsPerInch())
    # print("Physical DPI:", screen.physicalDotsPerInch())

    # screen_dpi = 100 # default 100
    # screen_dpi = 122.38 # 24'' 2560x1440 px screen
    screen_dpi = 92.60 # DELL U2414H 23.8'' FullHD
    # screen_dpi = 91.79 # 24'' FHD
    # screen_dpi = 165.63 # 13.3'' FHD

    return screen_dpi

from mplstyles import configure_mpl