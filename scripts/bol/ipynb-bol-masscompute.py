# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>
# <codecell>

# %matplotlib notebook

import os
import numpy as np
import matplotlib.pyplot as plt

from g2ltk import datareading, datasaving, utility, rivuletfinding

utility.configure_mpl()


# <codecell>

### Datasets display
datareading.set_default_root_path('../')
datareading.describe_root_path()


# <codecell>

### Dataset selection & acquisitions display
dataset = 'meandersspeed_zoom'

datareading.describe_dataset(dataset=dataset, videotype='gcv', makeitshort=True)
dataset_path = datareading.generate_dataset_path(dataset)


# <codecell>

for acquisition in ['m120', 'm130', 'm140', 'm150']:
# for acquisition in ['meandrage_clean_demo']:

    acquisition_path = os.path.join(dataset_path, acquisition)
    ### Parameters definition
    
    # conversion factor
    px_per_mm = 0.
    px_per_um = px_per_mm * 1e3
    
    # parameters to find the rivulet
    rivfinding_params = {
        'resize_factor': 2,
        'remove_median_bckgnd_zwise': True,
        'gaussian_blur_kernel_size': (1, 1), # (sigma_z, sigma_x)
        'white_tolerance': 70,
        'borders_min_distance': 5.,
        'borders_width': 6.,
        'max_rivulet_width': 150.,
    }
    
    # portion of the video that is of interest to us
    framenumbers = np.arange(datareading.get_number_of_available_frames(acquisition_path))
    roi = None, None, None, None  #start_x, start_y, end_x, end_y
    
    if dataset=='cleandemomeandrage':
        roi = None, 500, None, 700
    
    datareading.describe_acquisition(dataset, acquisition, subregion=roi)

    z_raw = datasaving.fetch_or_generate_data('bol', dataset, acquisition, framenumbers=framenumbers, roi=roi, **rivfinding_params)
    z_raw = datasaving.fetch_or_generate_data('fwhmol', dataset, acquisition, framenumbers=framenumbers, roi=roi, **rivfinding_params)


# <codecell>



