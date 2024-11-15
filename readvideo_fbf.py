#!/usr/bin/env python
import os
import cv2
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams["figure.max_open_warning"] = 50

plt.rcParams['pgf.texsystem'] = 'pdflatex'
plt.rcParams.update({'font.family': 'serif', 'font.size': 12,
                     'figure.titlesize' : 12,
                     'axes.labelsize': 12,'axes.titlesize': 12,
                     'legend.fontsize': 12})

from matplotlib.colors import Normalize # colormaps

from tools import datareading
#%%
# Datasets display
root_path = '../'
datasets = datareading.find_available_datasets(root_path)
print('Available datasets:', datareading.find_available_datasets(root_path))
#%%
# Dataset selection & acquisitions display
dataset = '-'
if len(datasets) == 1:
    dataset = datasets[0]
    datareading.log_info(f'Auto-selected dataset {dataset}')
dataset_path = os.path.join(root_path, dataset)
datareading.describe_dataset(dataset_path, type='gcv', makeitshort=True)
#%%
# Acquisition selection
acquisition = '10Hz_decal'
acquisition_path = os.path.join(dataset_path, acquisition)
datareading.is_this_a_video(acquisition_path)
#%%
relative_colorscale:bool = False
remove_median_bckgnd = True #remove the mediab img, practical for dirt on plate
median_correc = False # remove the median value over each z line. helps with the heterogenous lighting.
remove_bright_spots = False # removes bright spots by accepting cmap saturation (1%)

normalize_each_image = False
#%%
# Parameters definition
framenumbers = np.arange(datareading.get_number_of_available_frames(acquisition_path) or 1)
roi = None, None, None, None  #start_x, start_y, end_x, end_y
if acquisition=='drainagelent':
    roi = 800, 600, 1400, 900  #start_x, start_y, end_x, end_y
if (dataset, acquisition)==('Nalight_cleanplate_20240708', '10Hz_decal'):
    framenumbers = np.arange(1400, 1600)
#%%
# Data fetching
datareading.describe(dataset, acquisition, framenumbers=framenumbers, subregion=roi)
frames = datareading.get_frames(acquisition_path, framenumbers = framenumbers, subregion=roi)
length, height, width = frames.shape

acquisition_frequency = datareading.get_acquisition_frequency(acquisition_path, unit="Hz")
t = datareading.get_times(acquisition_path, framenumbers=framenumbers, unit='s')
#%%
# luminosity corrections
if remove_median_bckgnd:
    median_bckgnd = np.median(frames, axis=0, keepdims=True)
    frames = frames - median_bckgnd

if median_correc:
    frames = frames - np.median(frames, axis=(0,1), keepdims=True)

# if remove_median_bckgnd or median_correc:
#     frames -= frames.min()
#     frames *= 255/frames.max()

if normalize_each_image:
    # frames.setflags(write=1)
    # im = np.zeros_like(frames[0])
    # for i in range(length):
    #     cv2.normalize(frames[i], im, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    #     frames[i] = im
    frames = frames - frames.min(axis = (1,2), keepdims=True)
    # frames *= 255/frames.max(axis=0, keepdims=True)

vmin_absolutecmap = frames.min()
vmax_absolutecmap = frames.max()
if remove_bright_spots:
    vmin_absolutecmap = np.percentile(frames.flatten(), 1)
    vmax_absolutecmap = np.percentile(frames.flatten(), 99)
#%%
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
    global frames, t, length
    i = i % length
    s = t[i]%60
    m = t[i]//60
    ax.set_title(f't = {f"{m} m " if np.max(t[-1])//60 > 0 else ""}{s:.2f} s - frame {framenumbers[i]} ({i+1}/{length})')

    frame = frames[i]

    global see_image, median_correc, remove_bright_spots
    global img
    if see_image:
        image = frame.astype(float)
        img.set_array(image)
        if relative_colorscale:
            vmin_relativecmap = image.min()
            vmax_relativecmap = image.max()
            if remove_bright_spots:
                vmin_relativecmap = np.percentile(image.flatten(), 1)
                vmax_relativecmap = np.percentile(image.flatten(), 99)
            relative_norm = Normalize(vmin=vmin_relativecmap, vmax=vmax_relativecmap)
            img.set_norm(relative_norm)

    fig.canvas.draw()
#%%
### Display
fig, ax = plt.subplots(1, 1)  # initialise la figure
fig.suptitle(f'{acquisition} ({dataset})')
ax.set_xlim(0, width)
ax.set_ylim(0, height)
plt.tight_layout()

see_image:bool = True
i = 0 # time

### ANIMATED ELEMENTS

# frame
img = ax.imshow(np.zeros((height, width)), origin='lower', vmin = vmin_absolutecmap, vmax = vmax_absolutecmap
                # , aspect='auto'
                )
if not see_image:
    img.set_cmap('binary') # white background

# initialize
update_display()

# plt.colorbar()

fig.canvas.mpl_connect('key_press_event', on_press)

plt.show()