# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>
# <codecell>

# %matplotlib notebook

import numpy as np
import matplotlib.pyplot as plt

from g2ltk import datareading, datasaving, utility, logging

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

acquisitions:utility.Dict[str, utility.Tuple[float, float]] = {}
for acquisition in datareading.find_available_videos(dataset=dataset):
    acquisition_path = datareading.generate_acquisition_path(acquisition, dataset=dataset)
    acquisitions[acquisition] = datareading.get_monotonic_timebound(acquisition_path, unit='s')

tmin = min([acquisitions[acquisition][0] for acquisition in acquisitions])
tmax = max([acquisitions[acquisition][1] for acquisition in acquisitions])


# <codecell>

scaledata_folder = datareading.generate_dataset_path(dataset=dataset)

date = None
date = '2025-03-07'

date = date if date is not None else ''
scaledata_suffix = '.scale'
scaledata_acqus = [f[:-len(scaledata_suffix)] for f in os.listdir(scaledata_folder) if (f.endswith(scaledata_suffix) and (date in f))]
scaledata_acqus.sort()
print(scaledata_acqus)


# <markdowncell>

# * FIRST COLUMN : computer monotonic time (guerantees synchronicity with manta camera, understandable display because not too many numbers)
# * SECOND COLUMN : weight, in grams
# * THIRD COL : computer time (better precision than monotonic


# <codecell>

### ETAPE 1 LECTURE DU FICHIER

mono_time = np.array([], dtype=float)
weight = np.array([], dtype=float)
comp_time = np.array([], dtype=float)

for scaledata_acquisition in scaledata_acqus:

    filename = os.path.join(scaledata_folder, scaledata_acquisition+'.scale')

    with open(filename, "r") as f:
        num_lines = sum(1 for _ in f)

    scaledata = np.empty([num_lines, 3], dtype=float)


    with open(filename, "r") as f:
        for i_line, line in enumerate(f):
            for i_col in range(3):
                scaledata[i_line] = np.array( line[:-1].split('\t') ).astype(float)

    mono_time = np.concatenate((mono_time, scaledata[:, 0]))
    weight = np.concatenate((weight, scaledata[:, 1]))
    comp_time = np.concatenate((comp_time, scaledata[:, 2]))

t_s = mono_time.copy() / 1000
m = weight.copy()


# <codecell>

plt.figure(figsize=utility.figsize('double'))
ax = plt.gca()

ax.plot(t_s, m)
ax.set_xlabel('t [s]')
ax.set_ylabel('m [g]')

for acquisition in acquisitions:
    ax.axvspan(acquisitions[acquisition][0], acquisitions[acquisition][1], color='g', alpha=.2, lw=0)
    ax.text((acquisitions[acquisition][0]+acquisitions[acquisition][1])/2, (ax.get_ylim()[0]+ax.get_ylim()[1])/2, acquisition, ha='center', va='center')
    
ax.set_xlim(tmin - 100, tmax + 100)


# <codecell>

### STEP 2 : find and remove the flushes

dmdt = np.gradient(weight, t_s)

flush_threshold = -3 # every flux below this will be considered a flush
Dt_before_s = 5 # we discard point a bit before the flush
Dt_after_s = 5 + 1  # we discard point a bit after the flush

i_to_remove = np.where(dmdt < flush_threshold)[0]
to_remove = np.zeros_like(t_s, dtype=bool)

for i in i_to_remove:
    t_to_remove = t_s[i]
    to_remove = to_remove | ((t_s > t_to_remove - Dt_before_s) & (t_s < t_to_remove + Dt_after_s))

t_flush_start = t_s[:-1][to_remove[1:] & (~to_remove[:-1])] + .5 * utility.step(t_s)
t_flush_end = t_s[:-1][to_remove[:-1] & (~to_remove[1:])] + .5 * utility.step(t_s)

t_clean_s = t_s[~to_remove]
m_clean = m[~to_remove]
dmdt_clean = dmdt[~to_remove]


# <codecell>

fig, axes = plt.subplots(2, 1, squeeze=False, figsize=utility.figsize('double'), sharex=True)

ax = axes[0,0]
ax.plot(t_s, weight, color='gray')
for i_flush in range(len(t_flush_start)):
    t_start = t_flush_start[i_flush]
    t_end = t_flush_end[i_flush] if i_flush < len(t_flush_start) else t_s[-1]
    ax.axvspan(t_start, t_end, alpha=0.6, color='r', lw=0)
ax.plot(t_clean_s, m_clean, color='k')

ax = axes[1,0]
for i_flush in range(len(t_flush_start)):
    t_start = t_flush_start[i_flush]
    t_end = t_flush_end[i_flush] if i_flush < len(t_flush_start) else t_s[-1]
    ax.axvspan(t_start, t_end, alpha=0.6, color='r', lw=0)
ax.plot(t_s, dmdt, color='gray')
ax.plot(t_clean_s, dmdt_clean, color='k')


for acquisition in acquisitions:
    ax.axvspan(acquisitions[acquisition][0], acquisitions[acquisition][1], color='g', alpha=.2, lw=0)
    ax.text((acquisitions[acquisition][0]+acquisitions[acquisition][1])/2, (ax.get_ylim()[0]+ax.get_ylim()[1])/2, acquisition, ha='center', va='center')
ax.set_xlim(tmin - 100, tmax + 100)


# <codecell>

### Step 3 : gaussian filtering
# We want, say, one point per second
dt_smooth = 1.
t_smooth_s = np.arange(int(t_clean_s.min()), int(t_clean_s.max())+1 + dt_smooth/2, dt_smooth)[1:-1]


# we gaussiannise on a lenght of sigma
sigma_s = 15.

dmdt_smooth = np.empty(len(t_smooth_s), dtype=float)
for i_t_smooth, t_smooth in enumerate(t_smooth_s):
    weights = utility.gaussian_unnormalized(t_clean_s, mu=t_smooth, sigma=sigma_s)
    dmdt_smooth[i_t_smooth] = np.sum(weights * dmdt_clean) / np.sum(weights)


# <codecell>

fig, axes = plt.subplots(1, 1, squeeze=False, figsize=utility.figsize('double'))

ax = axes[0,0]
for i_flush in range(len(t_flush_start)):
    t_start = t_flush_start[i_flush]
    t_end = t_flush_end[i_flush] if i_flush < len(t_flush_start) else t_s[-1]
    ax.axvspan(t_start, t_end, alpha=0.6, color='r', lw=0)
# ax.plot(t_clean_s, m_clean, color='k', alpha=0.1)
ax.plot(t_smooth_s, dmdt_smooth, color='k')
ax.set_ylim(0, ax.get_ylim()[1])


for acquisition in acquisitions:
    ax.axvspan(acquisitions[acquisition][0], acquisitions[acquisition][1], color='g', alpha=.2, lw=0)
    ax.text((acquisitions[acquisition][0]+acquisitions[acquisition][1])/2, (ax.get_ylim()[0]+ax.get_ylim()[1])/2, acquisition, ha='center', va='center')
ax.set_xlim(tmin - 100, tmax + 100)

