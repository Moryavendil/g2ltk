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
root_path = '../'
datasets = datareading.find_available_datasets(root_path)
print('Available datasets:', datareading.find_available_datasets(root_path))


# <codecell>

### Dataset selection & acquisitions display
dataset = '-'
if len(datasets) == 1:
    dataset = datasets[0]
    datareading.log_info(f'Auto-selected dataset {dataset}')
dataset_path = os.path.join(root_path, dataset)
datareading.describe_dataset(dataset_path, type='gcv', makeitshort=True)


# <codecell>

utility.set_verbose('debug')


# <codecell>

for acquisition in ['a120', 'a340', 'a130', 'a140', 'a150', 'a160', 'a210', 'a220', 'a240', 'a280',
                    'a170', 'a180', 'a190', 'a200', 'a250', 'a260', 'a270', 'a290', 'a300', 
                    'a310', 'a320', 'a330', 'a350_2', 'a230', 'a350']:

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

    z_raw = datasaving.fetch_or_generate_data('bol', dataset, acquisition, framenumbers=framenumbers, roi=roi, **rivfinding_params)
    z_raw = datasaving.fetch_or_generate_data('fwhmol', dataset, acquisition, framenumbers=framenumbers, roi=roi, **rivfinding_params)


# <codecell>



