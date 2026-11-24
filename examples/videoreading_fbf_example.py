"""
Browsing a video frame by frame (fbf) with g2ltk.

This script opens one of the reference test videos and displays it one frame at a time, in a window where the keyboard
moves through the video:
    right / left            next / previous frame
    shift + right / left    10 frames forward / backward
    up / down               100 frames forward / backward

The frames are read one at a time, when they are displayed: this also works for long videos, which do not fit in memory.

Run it with `python examples/videoreading_fbf_example.py`, from any folder.
See videoreading_example.ipynb for an introduction to reading videos with g2ltk.
"""
import os
import numpy as np
import matplotlib.pyplot as plt

from g2ltk import videoreading as vr
vr.set_verbose('info')

# %% Selecting the video

# The *root path* indicates where we are going to find the datasets (directories containing videos).
# Here it is the folder above this script, so that the script works wherever it is run from.
examples_dir = os.path.dirname(os.path.abspath(__file__))
vr.set_default_root_path(os.path.dirname(examples_dir))

# A *dataset* is the name of a directory containing videos (*acquisitions*) of interest
dataset = 'test_data'
vr.describe_dataset(dataset=dataset, makeitshort=True)

# An *acquisition* is the name of a video file (without extension)
acquisition = 'test_gcv'  # 'test_gcv', 'test_mp4' or 'test_mkv'
if not vr.is_this_a_video(acquisition=acquisition, dataset=dataset):
    raise FileNotFoundError(f'No video named {acquisition} in dataset {dataset}')

# The path of the video, given to the reading functions
acquisition_path = vr.generate_acquisition_path(acquisition, dataset=dataset)

# %% Choosing what to display

framenumbers = None  # the frames to browse, e.g. np.arange(10, 20), or None for all of them
roi = None, None, None, None  # region of interest: start_x, start_y, end_x, end_y (None: up to the edge)

vr.describe_acquisition(acquisition_path, framenumbers=framenumbers, subregion=roi)

framenumbers = vr.format_framenumbers(acquisition_path, framenumbers)  # None becomes the array of all the frames
t_s = vr.get_t_s(acquisition_path, framenumbers=framenumbers)  # time of each frame, in s

# Colorscale: absolute (0 to 255, the range of 8-bit videos), or relative (adjusted to each frame)
relative_colorscale = True
saturate_a_bit = True  # with a relative colorscale, saturate the 1 % darkest and the 1 % brightest pixels

# %% Display

# Keys moving through the video, and by how many frames
STEPS = {'right': 1, 'left': -1, 'shift+right': 10, 'shift+left': -10, 'up': 100, 'down': -100}

# By default, matplotlib undoes and redoes zooms with the left and right keys: free them for the navigation
for keymap in ['keymap.back', 'keymap.forward']:
    plt.rcParams[keymap] = [key for key in plt.rcParams[keymap] if key not in STEPS]

fig, ax = plt.subplots(figsize=(6, 6))
fig.suptitle(f'{acquisition} ({dataset})')
image = ax.imshow(vr.get_frame(acquisition_path, framenumbers[0], subregion=roi), cmap='gray', vmin=0, vmax=255)
ax.set_xlabel('x [px]')
ax.set_ylabel('z [px]')

i = 0  # position of the displayed frame in framenumbers

def show_frame():
    frame = vr.get_frame(acquisition_path, framenumbers[i], subregion=roi)
    image.set_data(frame)
    if relative_colorscale:
        if saturate_a_bit:
            image.set_clim(*np.percentile(frame, [1, 99]))
        else:
            image.set_clim(frame.min(), frame.max())
    ax.set_title(f't = {t_s[i]:.3f} s - frame {framenumbers[i]} ({i + 1}/{len(framenumbers)})')
    fig.canvas.draw_idle()

def on_key_press(event):
    global i
    if event.key in STEPS:
        i = (i + STEPS[event.key]) % len(framenumbers)  # past the last frame, back to the first
        show_frame()

show_frame()
fig.canvas.mpl_connect('key_press_event', on_key_press)

print('Click on the figure, then use the keys right / left (1 frame), shift + right / left (10 frames) '
      'and up / down (100 frames) to browse the video.')
plt.show()
