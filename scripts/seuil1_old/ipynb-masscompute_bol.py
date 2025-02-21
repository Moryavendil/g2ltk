# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>
# <codecell>

%matplotlib notebook

import os
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams["figure.max_open_warning"] = 50

plt.rcParams['pgf.texsystem'] = 'pdflatex'
plt.rcParams.update({'font.family': 'serif', 'font.size': 12,
                     'figure.titlesize' : 12,
                     'axes.labelsize': 12,'axes.titlesize': 12,
                     'legend.fontsize': 12})

from tools import datareading, rivuletfinding, datasaving, utility


# <codecell>

# Dataset selection
dataset = '20241104'
dataset_path = os.path.join('../', dataset)
print('Available acquisitions:', datareading.find_available_gcv(dataset_path))


# <codecell>

for acquisition in ['100eta_gcv', '100mid_gcv', '100seuil_gcv', '200eta_gcv', '200mid_gcv', '200seuil_gcv', '20break_gcv', '20down_gcv', '20eta_gcv', '20regbreak_gcv', '20se_gcv', '20seuil_gcv', '30down_gcv', '30eta_gcv', '30max_gcv', '30seuil_gcv', '40down_gcv', '40eta_gcv', '40fix_gcv', '40seuil_gcv', '50eta_gcv', '50high_gcv', '50mid_gcv', '50seuil_gcv', '70eta_gcv', '70mid_gcv', '70seuil_gcv', 'rest_gcv', 'scalefrontthenback_gcv']:

    acquisition_path = os.path.join(dataset_path, acquisition)

    # Parameters definition
    rivfinding_params = {
        'resize_factor': 2,
        'borders_min_distance': 8,
        'max_borders_luminosity_difference': 50,
        'max_rivulet_width': 100.,
    }
    framenumbers = np.arange(datareading.get_number_of_available_frames(acquisition_path))
    roi = None, None, None, None  #start_x, start_y, end_x, end_y
    
    # framenumbers = np.arange(100)
    roi = 250, None, 1150, None
    
    datareading.describe(dataset, acquisition, verbose=3)
    # Data fetching
    length, height, width = datareading.get_geometry(acquisition_path, framenumbers = framenumbers, subregion=roi)
    
    t = datareading.get_t_frames(acquisition_path, framenumbers=framenumbers)
    x = datareading.get_x_px(acquisition_path, framenumbers = framenumbers, subregion=roi, resize_factor=rivfinding_params['resize_factor'])
    
    z_raw = datasaving.fetch_or_generate_data('bol', dataset, acquisition, framenumbers=framenumbers, roi=roi, **rivfinding_params)
    w_raw = utility.w_form_borders(datasaving.fetch_or_generate_data('borders', dataset, acquisition, framenumbers=framenumbers, roi=roi, **rivfinding_params))

