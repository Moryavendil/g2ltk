# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>
# <codecell>

# %matplotlib qt
# import matplotlib ; matplotlib.use('Qt5Agg') # use this line if in a Python script

import os
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.colors import Normalize # colormaps

from g2ltk import set_verbose, datareading, datasaving, utility, rivuletfinding

utility.configure_mpl()

### Datasets display
datareading.set_default_root_path('../')
datareading.describe_root_path()


# <codecell>

### Dataset selection & acquisitions display
dataset = 'meandersspeed_zoom'

datareading.describe_dataset(dataset=dataset, videotype='gcv', makeitshort=True)
dataset_path = datareading.generate_dataset_path(dataset)


# <codecell>

# Acquisition selection
acquisition = 'ha_n140f500a1000_gcv'
acquisition_path = os.path.join(dataset_path, acquisition)
datareading.is_this_a_video(acquisition_path)


# <codecell>

# Parameters definition
framenumbers = datareading.format_framenumbers(acquisition_path, None)
roi = None, None, None, None  #start_x, start_y, end_x, end_y
# Data fetching
datareading.describe_acquisition(dataset, acquisition, framenumbers=framenumbers, subregion=roi)
length, height, width = datareading.get_geometry(acquisition_path, framenumbers=framenumbers, subregion=roi)
acquisition_frequency = datareading.get_acquisition_frequency(acquisition_path, unit="Hz")
t = datareading.get_times(acquisition_path, framenumbers=framenumbers, unit='s')
# Colorscale option
relative_colorscale: bool = True
saturate_a_bit: bool = True

vmin_absolutecmap = 0
vmax_absolutecmap = 255
if dataset == 'illustrations' and acquisition == '1200_s_break_gcv':
    vmax_absolutecmap = 10
if dataset == 'illustrations' and acquisition == '300_s_break_gcv':
    vmax_absolutecmap = 10
if dataset == 'illustrations' and acquisition == '600_s_break_gcv':
    vmax_absolutecmap = 10
if dataset == 'illustrations' and acquisition == '60_s_break_gcv':
    vmax_absolutecmap = 70


# <codecell>

def on_press(event):
    # print('press', event.key)
    global height, width
    # Navigation
    global i
    if event.key == 'right':
        i += 1
    if event.key == 'left':
        i -= 1
    if event.key == 'shift+right':
        i += 10
    if event.key == 'shift+left':
        i -= 10
    if event.key == 'up':
        i += 100
    if event.key == 'down':
        i -= 100

    update_display()

def update_display():
    global i, fig
    global t, length
    i = i % length
    ax.set_title(f't = {utility.format_videotime(t[i], finaltime_s=t[-1])} - frame {framenumbers[i]} ({i+1}/{length})')

    frame = datareading.get_frame(acquisition_path, i, subregion=roi)
    if dataset == 'illustrations' and acquisition == '1200_s_break_gcv':
        frame[11, :] = frame[10, :]

    global see_image, median_correc, saturate_a_bit
    global img
    if see_image:
        image = frame.astype(float)
        img.set_array(image)
        if relative_colorscale:
            vmin_relativecmap = image.min()
            vmax_relativecmap = image.max()
            if saturate_a_bit:
                vmin_relativecmap = np.percentile(image.flatten(), 1)
                vmax_relativecmap = np.percentile(image.flatten(), 99)
            relative_norm = Normalize(vmin=vmin_relativecmap, vmax=vmax_relativecmap)
            img.set_norm(relative_norm)

    fig.canvas.draw()


# <codecell>

### Display
fig, ax = plt.subplots(1, 1)  # initialise la figure
fig.suptitle(f'{acquisition} ({dataset})')
ax.set_xlim(-.5, width + .5)
ax.set_ylim(-.5, height + .5)
plt.tight_layout()

see_image:bool = True
i = 0 # time

### ANIMATED ELEMENTS

# frame
img = ax.imshow(datareading.get_frame(acquisition_path, i, subregion=roi), origin='lower', vmin = vmin_absolutecmap, vmax = vmax_absolutecmap
                # , aspect='auto'
                )
if not see_image:
    img.set_cmap('binary') # white background

# initialize
update_display()

# plt.colorbar()

fig.canvas.mpl_connect('key_press_event', on_press)

plt.show()

    

