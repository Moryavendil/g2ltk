# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>
# <codecell>

# %matplotlib notebook

import numpy as np
import matplotlib.pyplot as plt

from g2ltk import datareading, datasaving, utility

utility.configure_mpl()


# <codecell>

### Datasets display
datareading.set_default_root_path('../')
datareading.describe_root_path()


# <codecell>

### Dataset selection & acquisitions display
dataset = datareading.find_dataset(None)
datareading.describe_dataset(dataset=dataset, videotype='gcv', makeitshort=True)


# <codecell>

for acquisition in []:

    acquisition_path = datareading.generate_acquisition_path(acquisition, dataset=dataset)

    #%%
    
    # Parameters definition
    # =====================
    
    # Data gathering
    # --------------
    
    ### portion of the video that is of interest to us
    framenumbers = np.arange(datareading.get_number_of_available_frames(acquisition_path))
    roi = None, None, None, None  #start_x, start_y, end_x, end_y
    
    # Rivulet detection
    # -----------------
    
    ### parameters to find the rivulet
    rivfinding_params = {
        'resize_factor': 2,
        'remove_median_bckgnd_zwise': True,
        'gaussian_blur_kernel_size': (1, 1), # (sigma_z, sigma_x)
        'white_tolerance': 70,
        'borders_min_distance': 5.,
        'borders_width': 6.,
        'max_rivulet_width': 150.,
    }
    
    #%%
    
    datareading.describe_acquisition(dataset, acquisition, subregion=roi)

    #%%
    
    z_raw = datasaving.fetch_or_generate_data('bol', dataset, acquisition, framenumbers=framenumbers, roi=roi, **rivfinding_params)
    w_raw = datasaving.fetch_or_generate_data('fwhmol', dataset, acquisition, framenumbers=framenumbers, roi=roi, **rivfinding_params)
    
utility.log_info('TERMINÉ')

