# %matplotlib notebook

import os
import numpy as np
import matplotlib.pyplot as plt

# plt.rcParams['pgf.texsystem'] = 'pdflatex'
# plt.rcParams.update({'font.family': 'serif', 'font.size': 18,
#                      'figure.titlesize' : 28,
#                      'axes.labelsize': 20,'axes.titlesize': 24,
#                      'legend.fontsize': 20, 'legend.handlelength': 2})
# plt.rcParams['text.usetex'] = True
plt.rcParams["figure.figsize"] = (12, 8)

from matplotlib.colors import Normalize # colormaps
from scipy.optimize import curve_fit # curve fitting
from scipy.signal.windows import hann, flattop, boxcar, blackmanharris, tukey # FFT windowing

from tools import datareading, rivuletfinding, utility, datasaving

# Dataset selection
dataset = 'f50_d95_110'
dataset_path = os.path.join('../', dataset)
print('Available acquisitions:', datareading.find_available_videos(dataset_path))

acquisitions = datareading.find_available_videos(dataset_path)

import re

text = 'f50d95_3500'
try:
    found = re.search('d(.+?)_', text).group(1).zfill(3) + re.search('_(.+)', text).group(1)
except AttributeError:
    found = ''
print(found)

acquisitions.sort(key=lambda acqu: re.search('d(.+?)_', acqu).group(1).zfill(3) + re.search('_(.+)', acqu).group(1))

print(acquisitions)

for acquisition in acquisitions:
    # Acquisition selection
    acquisition_path = os.path.join(dataset_path, acquisition)
    datareading.is_this_a_video(acquisition_path)

    # Parameters definition
    framenumbers = None
    roi = None, None, None, None  #start_x, start_y, end_x, end_y

    if dataset == 'ramp':
        roi = None, 120, None, 300  #start_x, start_y, end_x, end_y

    # Data fetching
    frames = datareading.get_frames(acquisition_path, framenumbers = framenumbers, subregion=roi)
    length, height, width = frames.shape

    t = datareading.get_times(acquisition_path, framenumbers=framenumbers, unit='s')

    acquisition_frequency = datareading.get_acquisition_frequency(acquisition_path, unit="Hz")

    # print(f'Dataset: "{dataset}", acquisition: "{acquisition}"')
    # print(f'Frames dimension: {height}x{width}')
    # print(f'Length: {length} frames ({round(datareading.get_acquisition_duration(acquisition_path, framenumbers=framenumbers, unit="s"), 2)} s - {frames.nbytes/10**6} MB)')
    # print(f'Acquisition frequency: {round(datareading.get_acquisition_frequency(acquisition_path, unit="Hz"), 2)} Hz')

    print(f'Dataset: "{dataset}", acquisition: "{acquisition}"')

    rivs = datasaving.fetch_or_generate_data('cos', dataset, acquisition, framenumbers=framenumbers, roi=roi)
    try:
        brds = datasaving.fetch_or_generate_data('borders', dataset, acquisition, framenumbers=framenumbers,
                                                 roi=roi)
    except:
        pass

    try:
        bol = datasaving.fetch_or_generate_data('bol', dataset, acquisition, framenumbers=framenumbers, roi=roi)
    except:
        pass

    print('DONE')

