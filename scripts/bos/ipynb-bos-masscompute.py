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

for acquisition in ['highQ_f0250a1200', 'highQ_f0333a1200', 'highQ_f0500a1200', 'highQ_f0660a1200', 'highQ_f1000a1200', 'highQ_f2000a1200', 'highQ_f3000a1200', 'highQ_f4000a1200', 'highQ_f5000a1200', 'highQ_naturel', 'naturel', 'naturel2', 'f0500a1200', 'f0750a1200', 'f1000a1200', 'f1500a1200']:
# for acquisition in ['naturel']:

    acquisition_path = os.path.join(dataset_path, acquisition)

    # parameters to find the rivulet
    rivfinding_params = {
        'resize_factor': 2,
        'white_tolerance': 70,
        'rivulet_size_factor': 2.,
        'remove_median_bckgnd_zwise': True,
        'gaussian_blur_kernel_size': (1, 1), # (sigma_z, sigma_x)
    }
    
    # portion of the video that is of interest to us
    framenumbers = np.arange(datareading.get_number_of_available_frames(acquisition_path))
    roi = None, None, None, None  #start_x, start_y, end_x, end_y
    
    if dataset=='250225':
        roi = [None, 10, 1980, -10]
    
    z_raw = datasaving.fetch_or_generate_data('cos', dataset, acquisition, framenumbers=framenumbers, roi=roi, **rivfinding_params)


# <codecell>



