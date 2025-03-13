# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>
# <codecell>

# %matplotlib qt
# import matplotlib ; matplotlib.use('Qt5Agg') # use this line if in a Python script

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.colors import Normalize # colormaps

from g2ltk import datareading, datasaving, utility, rivuletfinding

utility.configure_mpl()

### Datasets display
datareading.set_default_root_path('../')
datareading.describe_root_path()


# <codecell>

### Dataset selection & acquisitions display
dataset = datareading.find_dataset(None)
datareading.describe_dataset(dataset=dataset, videotype='gcv', makeitshort=True)


# <codecell>

### Acquisition selection
acquisition = 'q60_gcv'
acquisition_path = datareading.generate_acquisition_path(acquisition, dataset=dataset)


# <codecell>

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


# <codecell>

compute_before = False


# <codecell>

# Data fetching
datareading.describe_acquisition(dataset, acquisition, framenumbers = framenumbers, subregion=roi)

length, height, width = datareading.get_geometry(acquisition_path, framenumbers = framenumbers, subregion=roi)

t = datareading.get_t_frames(acquisition_path, framenumbers=framenumbers)
x = datareading.get_x_px(acquisition_path, framenumbers = framenumbers, subregion=roi, resize_factor=rivfinding_params['resize_factor'])

if compute_before:
    borders = datasaving.fetch_or_generate_data('borders', dataset, acquisition, framenumbers=framenumbers, roi=roi, **rivfinding_params)
    z_raw = datasaving.fetch_or_generate_data('bol', dataset, acquisition, framenumbers=framenumbers, roi=roi, **rivfinding_params)
    w_raw = datasaving.fetch_or_generate_data('fwhmol', dataset, acquisition, framenumbers=framenumbers, roi=roi, **rivfinding_params)


# <codecell>

# Real-unit data
acquisition_frequency = datareading.get_acquisition_frequency(acquisition_path, unit="Hz")
t_s = datareading.get_t_s(acquisition_path, framenumbers=framenumbers)


# <codecell>

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

plt.rcParams["keymap.back"] = ['c']
plt.rcParams["keymap.forward"] = ['v']

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
    global t_s, length
    i = i % length
    ax_img.set_title(f't = {utility.format_videotime(t[i], finaltime_s=t[-1])} s - frame {framenumbers[i]} ({i + 1}/{length})')

    frame = datareading.get_frame(acquisition_path, i, subregion=roi)

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
            
    borders = rivuletfinding.borders_framewise(frame, **rivfinding_params)
    z = rivuletfinding.bol_framewise(frame, borders_for_this_frame=borders, **rivfinding_params)
    w = rivuletfinding.fwhmol_framewise(frame, borders_for_this_frame=borders, **rivfinding_params)

    img_z.set_data(x, z)
    img_w_bot.set_data(x, z-w/2)
    img_w_top.set_data(x, z+w/2)
    img_border_bot.set_data(x, borders[0])
    img_border_top.set_data(x, borders[1])
    
    riv_z.set_data(x, z)
    
    riv_w.set_data(x, w)
    if not compute_before:
        if np.max(z) > ax_z.get_ylim()[1]:
            ax_z.set_ylim(ax_z.get_ylim()[0], z.mean()+(z.max()-z.mean())*1.25)
        if np.min(z) < ax_z.get_ylim()[0]:
            ax_z.set_ylim(z.mean()+(z.min()-z.mean())*1.25, ax_z.get_ylim()[1])
        if np.max(w) > ax_w.get_ylim()[1]:
            ax_w.set_ylim(0, w.max()*1.25)

    fig.canvas.draw()


# <codecell>

### Display
fig, axes = plt.subplots(3, 1, sharex=True)  # initialise la figure
fig.suptitle(f'{acquisition} ({dataset})')

ax_img = axes[0]
ax_img.set_xlim(-.5, width + .5)
ax_img.set_ylim(-.5, height + .5)


see_image:bool = True
i = 0 # time

### ANIMATED ELEMENTS

# frame
frame1 = datareading.get_frame(acquisition_path, i, subregion=roi)  
img = ax_img.imshow(frame1, origin='lower', vmin = vmin_absolutecmap, vmax = vmax_absolutecmap
                  # , aspect='auto'
                  )
if not see_image:
    img.set_cmap('binary') # white background
img_z, = ax_img.plot([], [], color='k')
img_w_bot, = ax_img.plot([], [], color='w', ls='--')
img_w_top, = ax_img.plot([], [], color='w', ls='--')
img_border_bot, = ax_img.plot([], [], color='gray', ls=':')
img_border_top, = ax_img.plot([], [], color='gray', ls=':')

ax_z = axes[1]
z1 = rivuletfinding.bol_framewise(frame1, **rivfinding_params)
ax_z.set_ylim(z1.mean()-1, z1.mean()+1)
riv_z, = ax_z.plot([], [], color='k')

ax_w = axes[2]
ax_w.set_ylim(0, 1)
riv_w, = ax_w.plot([], [], color='k')

# initialize
update_display()

# plt.colorbar()

fig.canvas.mpl_connect('key_press_event', on_press)

# plt.tight_layout()
plt.show()

    

