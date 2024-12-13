# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>
# <codecell>

from tools.utility import activate_saveplot
%matplotlib notebook

import os
import numpy as np
import matplotlib.pyplot as plt

from tools import datareading, utility
utility.configure_mpl()


# <codecell>

# Datasets display
root_path = '../'
datasets = datareading.find_available_datasets(root_path)
print('Available datasets:', datareading.find_available_datasets(root_path))


# <codecell>

# Dataset selection & acquisitions display
dataset = 'Nalight_cleanplate_20240708'
if len(datasets) == 1:
    dataset = datasets[0]
    datareading.log_info(f'Auto-selected dataset {dataset}')
dataset_path = os.path.join(root_path, dataset)
datareading.describe_dataset(dataset_path, type='gcv', makeitshort=True)


# <codecell>

# Acquisition selection
acquisition = '1Hz_strong'
acquisition_path = os.path.join(dataset_path, acquisition)


# <codecell>

# Parameters definition
framenumbers = np.arange(datareading.get_number_of_available_frames(acquisition_path))
roi = None, None, None, None  #start_x, start_y, end_x, end_y

framenumbers = [1094, 1115, 1194, 1215]


# <codecell>

# Data fetching
datareading.describe_acquisition(dataset, acquisition, framenumbers = framenumbers, subregion=roi)
frames = datareading.get_frames(acquisition_path, framenumbers = framenumbers, subregion=roi)
length, height, width = frames.shape

acquisition_frequency = datareading.get_acquisition_frequency(acquisition_path, unit="Hz")
t = datareading.get_times(acquisition_path, framenumbers=framenumbers, unit='s')


# <codecell>

um_per_px = None
if dataset=='Nalight_cleanplate_20240708':
    um_per_px = 5000 / 888
mm_per_px = um_per_px/1000 if um_per_px is not None else None

density = 1.72
refractive_index = 1.26
lambda_Na_void = 0.589 # in um
gamma = 14
bsur2 = 600/2 # in mm

lambd = lambda_Na_void/refractive_index # in um
h0 = lambd * np.pi / (2 * np.pi)/2 # h0 = lambd/4 in um

from matplotlib.colors import LinearSegmentedColormap
cmap_Na = LinearSegmentedColormap.from_list("cmap_Na", ['black', "xkcd:bright yellow"])

vmin_Na = np.percentile(frames, 1)
vmax_Na = np.percentile(frames, 99)


# <codecell>

frame = frames[0]

from scipy.ndimage import gaussian_filter

x_px = np.arange(frame.shape[1])
z_px = np.arange(frame.shape[0])

X_px, Z_px = np.meshgrid(x_px, z_px)

# real interesting zone
interest = (Z_px > 70 + 115/X_px.max()*X_px) & (Z_px < 805 - 70/X_px.max()*X_px)

# easy square zone
interest = (Z_px > 70 + 115) & (Z_px < 805 - 70)

sgma = (.5, .5)
frame_intrest = gaussian_filter(frame, sigma=sgma)[interest].reshape((interest.sum()//frame.shape[1], frame.shape[1]))

img = frame_intrest.copy()

vmi = np.percentile(img, 0.1, axis=1, keepdims=True)
vma = np.percentile(img, 99.9, axis=1, keepdims=True)

img = (img - vmi)/(vma-vmi)
img[img < 0] = 0
img[img > 1] = 1
img = img*2 -1 


# <codecell>

fig, axes = plt.subplots(2, 1)

ax = axes[0]
ax.imshow(frame*(interest*.25+.75), cmap=cmap_Na, vmin=vmin_Na, vmax=vmax_Na, aspect='auto')

ax = axes[1]
ax.imshow(img, cmap='bwr', vmin=-1, vmax=1, aspect='auto')


# <codecell>

from scipy.signal import hilbert2


# <codecell>

zc = 290

img_top = img[:zc]
img_bot = img[zc:]

### TOP
sig = img_top
hh, ww = sig.shape
sig_x4 = np.zeros((2 * hh, 2 * ww), dtype=sig.dtype)
sig_x4[:hh, :ww] = sig[:, :]
sig_x4[hh:, :ww] = sig[::-1, :]
sig_x4[:hh, ww:] = sig[:, ::-1]
sig_x4[hh:, ww:] = sig[::-1, ::-1]
hil2D_x4 = hilbert2(sig_x4)
phase_wrapped_hil2D_x4 = np.angle(hil2D_x4)

phase_top = -phase_wrapped_hil2D_x4[:hh, ww:][:, ::-1]

### BOT
sig = img_bot
hh, ww = sig.shape
sig_x4 = np.zeros((2 * hh, 2 * ww), dtype=sig.dtype)
sig_x4[:hh, :ww] = sig[:, :]
sig_x4[hh:, :ww] = sig[::-1, :]
sig_x4[:hh, ww:] = sig[:, ::-1]
sig_x4[hh:, ww:] = sig[::-1, ::-1]
hil2D_x4 = hilbert2(sig_x4)
phase_wrapped_hil2D_x4 = np.angle(hil2D_x4)

phase_bot = +phase_wrapped_hil2D_x4[:hh, :ww]



# <codecell>

fig, axes = plt.subplots(2, 1, squeeze=False)

ax = axes[0,0]
ax.imshow(phase_top, aspect='auto')

ax = axes[1,0]
ax.imshow(phase_bot, aspect='auto')


# <codecell>

phase_wr = np.zeros(shape=img.shape, dtype=phase_wrapped_hil2D_x4.dtype)
phase_wr[:zc, :] = phase_top
phase_wr[zc:, :] = phase_bot

from skimage.restoration import unwrap_phase
phase_unwr = unwrap_phase(-phase_wr)


# <codecell>

fig, axes = plt.subplots(2)

ax = axes[0]
ax.imshow(phase_wr, aspect='auto')

ax = axes[1]
ax.imshow(phase_unwr, aspect='auto')


# <codecell>

X = X_px[interest].reshape((interest.sum()//frame.shape[1], frame.shape[1])) * mm_per_px
Z = Z_px[interest].reshape((interest.sum()//frame.shape[1], frame.shape[1])) * mm_per_px
H = phase_unwr  / (2 * np.pi) * lambd /2 + 5.5

H = gaussian_filter(H, 5)

X = X[:, ::-1]


# <codecell>

Z[zc]


# <codecell>

SAVEPLOT=False
utility.activate_saveplot(True)


# <codecell>

figw = utility.genfig.figw['wide']
fig, axes = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(figw, figw/1.3))
ax = axes

xlim = [X.min(), X.max()]
zlim = [Z.min(), Z.max()]

# Plot the surface.
# surf = ax.plot_surface(Z, X, H, cmap='inferno', vmin=0, vmax=6,
#                        lw=0, antialiased=False, alpha=.55, rcount=100, ccount=100)
surf = ax.plot_surface(Z, X, H, cmap='inferno', vmin=0, vmax=6,
                       lw=0, antialiased=True, alpha=.99, rcount=100, ccount=100)


ax.contour(Z, X, H, zdir='y', offset=xlim[1], colors='teal', levels=3)
# ax.text(2.75, xlim[1], 5.5, r'Transverse height profiles', 'x', color='teal', ha='center', va='bottom')

ax.plot(np.full_like(Z[zc], zlim[0]), X[zc], np.max(H, axis=0), color='darkred')
ax.plot(Z[zc], X[zc], np.max(H, axis=0), color='darkred')
ax.text(zlim[0], 5, 5, r'Crest height', (0., 1, 0.1), color='darkred', ha='center', va='bottom')

res = 10

facecolors = cmap_Na((frame_intrest-vmin_Na)/(vmax_Na-vmin_Na))

stepX = np.abs((X[0][1:] - X[0][:-1])).mean()
stepZ = np.abs((Z[:, 0][1:] - Z[:, 0][:-1])).mean()
# when no saveplot
# facecolors[Z > zlim[1] - stepZ*14] = (0,0,0,0)
# facecolors[X < X.min() + stepX*43] = (0,0,0,0)
# # when saveplot (too much)
# facecolors[Z > zlim[1] - stepZ*20] = (0,0,0,0)
# facecolors[X < X.min() + stepX*59] = (0,0,0,0)
# when saveplot (too much)
facecolors[Z > zlim[1] - stepZ*10] = (0,0,0,0)
facecolors[X < X.min() + stepX*35] = (0,0,0,0)

# image on bottom
ax.plot_surface(Z, X, np.full_like(H, 0), facecolors = facecolors, rcount=200, ccount=200)

ax.plot(Z[zc], X[zc], H[zc], color='darkred', zorder=10)

# for xc_mm in [2.5, 5, 7.5]:
#     xc = np.abs(X[0,:] - xc_mm).argmin()
#     ax.plot(Z[:, xc], X[:, xc], H[:, xc], color='teal', zorder=10)

ax.set_xlabel(r'$z$ [mm]')
ax.set_xlim(zlim)

ax.set_ylabel(r'$x$ [mm]')
ax.set_ylim(xlim)

ax.set_zlabel(r'Film height $h$ [$\mu$m]')
ax.set_zlim(0, 6.)

deltazxlim = (xlim[1]-xlim[0]) - (zlim[1]-zlim[0])
ax.set_xlim(zlim[0]-deltazxlim/2, zlim[1]+deltazxlim/2)
ax.set_zlim(0, 12)

# First remove fill
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
# Now set color to white (or whatever is "invisible")
ax.xaxis.pane.set_edgecolor('w')
ax.yaxis.pane.set_edgecolor('w')
ax.zaxis.pane.set_edgecolor('w')

ax.view_init(elev=20, azim=-60, roll=0)

if SAVEPLOT:
    utility.save_graphe('3dtunnel', bbox_inches='tight', pad_inches=.5, imageonly=True)


# <codecell>

plt.close()

