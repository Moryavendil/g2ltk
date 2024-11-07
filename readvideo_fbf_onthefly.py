import os
import cv2
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (12, 8)

from matplotlib.colors import Normalize # colormaps

from tools import datareading
#%%
# Dataset selection
dataset = '20241104'
dataset_path = '../' + dataset
print('Available acquisitions:', datareading.find_available_videos(dataset_path))
#%%
# Acquisition selection
acquisition = '100seuil_gcv'
acquisition_path = os.path.join(dataset_path, acquisition)
datareading.is_this_a_video(acquisition_path)
#%%
# see the frame
relative_colorscale:bool = False
#%%
# Parameters definition
framenumbers = np.arange(datareading.get_number_of_available_frames(acquisition_path))
roi = None, None, None, None  #start_x, start_y, end_x, end_y
if acquisition=='drainagelent':
    roi = 800, 600, 1400, 900  #start_x, start_y, end_x, end_y

nframes = datareading.get_number_of_available_frames(acquisition_path)
# nframes = min(nframes, 500)
framenumbers = np.arange(nframes)
#%%
# Data fetching
frametest = datareading.get_frame(acquisition_path, 0, subregion=roi)
height, width = frametest.shape

acquisition_frequency = datareading.get_acquisition_frequency(acquisition_path, unit="Hz")
t = datareading.get_times(acquisition_path, framenumbers=framenumbers, unit='s')
length = len(t)


print(f'Dataset: "{dataset}", acquisition: "{acquisition}"')
print(f'Frames dimension: {height}x{width}')
print(f'Length: {length} frames ({round(datareading.get_acquisition_duration(acquisition_path, framenumbers=framenumbers, unit="s"), 2)} s)')
print(f'Acquisition frequency: {round(datareading.get_acquisition_frequency(acquisition_path, unit="Hz"), 2)} Hz')
if (not(datareading.are_there_missing_frames(acquisition_path))):
    print(f'No dropped frames :)')
else:
    print(f'Dropped frames...')
#%%
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
    global t, length
    i = i % length
    s = t[i]%60
    m = t[i]//60
    ax.set_title(f't = {f"{m} m " if np.max(t[-1])//60 > 0 else ""}{s:.2f} s - frame {framenumbers[i]} ({i+1}/{length})')

    frame = datareading.get_frame(acquisition_path, i, subregion=roi)
    if dataset == 'illustrations' and acquisition == '1200_s_break_gcv':
        frame[11, :] = frame[10, :]

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