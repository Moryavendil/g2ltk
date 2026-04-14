# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>
# <codecell>

# %matplotlib notebook

import numpy as np
import matplotlib.pyplot as plt

from g2ltk import videoreading, logging
from g2ltk.rivulets import utility

utility.configure_mpl()


# <codecell>




# <codecell>

### Datasets display
videoreading.set_default_root_path('../')
videoreading.describe_root_path()


# <codecell>

### Dataset selection & acquisitions display
dataset = videoreading.find_dataset(None)
videoreading.describe_dataset(dataset=dataset, videotype='tiff', makeitshort=True)


# <codecell>

infotable = videoreading.obtain_metainfo(dataset)


# <codecell>

for acquisition in videoreading.find_available_videos(dataset=dataset, videotype='tiff'):

    acquisition_path = videoreading.generate_acquisition_path(acquisition, dataset=dataset)

    logging.log_debug(f'dataset: {dataset} | acquisition: {acquisition}')
    for colname in ['title', 'acquisition_frequency', 'exposure_time']:
        try:
            col = infotable[colname]
            logging.log_debug(f'Has column: {colname}')
        except KeyError as e:
            logging.log_error(f'Missing column: {e}')
    logging.log_debug(f"Found the acquisition in titles?: {(infotable['title'] == acquisition).max()}")

    acquisition_frequency = (float((infotable['acquisition_frequency'][infotable['title'] == acquisition]).iloc[0]))
    exposure_time = (float((infotable['exposure_time'][infotable['title'] == acquisition]).iloc[0]))

    logging.log_info(f'acquisition: {acquisition} | fps={acquisition_frequency} | texp={exposure_time}')

    # Parameters definition
    framenumbers = np.arange(videoreading.get_number_of_available_frames(acquisition_path))
    roi = None, None, None, None  #start_x, start_y, end_x, end_y

    videoreading.convert_tiff_to_gcv(acquisition_path, acquisition_frequency, exposure_time, framenumbers=framenumbers, subregion=roi)


# <markdowncell>

# ## TESTING
# We display the first, middle and last images of thegenerated gcv to check that eveything is OK


# <codecell>

### TEST NOW
acquisition_gcv = videoreading.find_available_videos(dataset=dataset, videotype='tiff')[0] + '_gcv'

acquisition_path = videoreading.generate_acquisition_path(acquisition_gcv, dataset=dataset)
videoreading.is_this_a_video(acquisition_path)


# <codecell>

# Parameters definition
mxfn = videoreading.get_number_of_available_frames(acquisition_path)
framenumbers = np.array([0, mxfn//2, mxfn-1])
roi = None, None, None, None  #start_x, start_y, end_x, end_y


# <codecell>

frames = videoreading.get_frames(acquisition_path, framenumbers = framenumbers, subregion=roi)
length, height, width = frames.shape


# <codecell>

fig, axes = plt.subplots(3, 1)
axes[0].imshow(frames[0])
axes[1].imshow(frames[1])
axes[2].imshow(frames[2])

