# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>
# <codecell>

%matplotlib notebook

import os
import numpy as np
import matplotlib.pyplot as plt

from tools import set_verbose, datareading, utility
utility.configure_mpl()


# <codecell>

# Datasets display
root_path = '../'
datasets = datareading.find_available_datasets(root_path)
print('Available datasets:', datareading.find_available_datasets(root_path))


# <codecell>

# Dataset selection & acquisitions display
dataset = '-'
if len(datasets) == 1:
    dataset = datasets[0]
    datareading.log_info(f'Auto-selected dataset {dataset}')
dataset_path = os.path.join(root_path, dataset)
datareading.describe_dataset(dataset_path, type='gcv', makeitshort=True)


# <codecell>

# Acquisition selection
# acquisition = 'ha_n140f001a500_gcv'
# acquisition = 'ha_n140f001a1000_gcv'
# acquisition = 'ha_n140f002a500_gcv'
# acquisition = 'ha_n140f002a800_gcv'
# acquisition = 'ha_n140f005a500_gcv'
acquisition = 'ha_n140f010a500_gcv'
# acquisition = 'ha_n140f020a500_gcv'
# acquisition = 'ha_n140f050a400_gcv'
# acquisition = 'ha_n140f100a300_gcv'
acquisition = 'ha_n140f200a400_gcv'
# acquisition = 'ha_n140f500a1000_gcv'
acquisition_path = os.path.join(dataset_path, acquisition)
datareading.is_this_a_video(acquisition_path)



# <codecell>

framenumbers = np.arange(datareading.get_number_of_available_frames(acquisition_path))[1:-1]
roi = None, None, None, None  #start_x, start_y, end_x, end_y

if acquisition == 'ha_n140f001a500_gcv':
    framenumbers = np.arange(182)

length, height, width = datareading.get_geometry(acquisition_path, framenumbers = framenumbers, subregion=roi)


# <codecell>

datareading.describe_acquisition(dataset, acquisition, framenumbers=framenumbers, subregion=roi)

frames = datareading.get_frames(acquisition_path, framenumbers=framenumbers, subregion=roi)
t = datareading.get_times(acquisition_path, framenumbers=framenumbers)
acquisition_frequency = datareading.get_acquisition_frequency(acquisition_path)


# <codecell>

row_of_interest = 80
row_avg = 5
cols_start = 300 # None
cols_stop = 500 # None


if acquisition == 'ha_n140f001a500_gcv':
    row_of_interest = 150
    row_avg = 5
    cols_start = 500 # None
    cols_stop = 900 # None
if acquisition == 'ha_n140f001a1000_gcv':
    row_of_interest = 150
    row_avg = 5
    cols_start = 500 # None
    cols_stop = 900 # None
if acquisition == 'ha_n140f002a500_gcv':
    row_of_interest = 150
    row_avg = 5
    cols_start = 500 # None
    cols_stop = 900 # None
if acquisition == 'ha_n140f002a800_gcv':
    row_of_interest = 150
    row_avg = 5
    cols_start = 500 # None
    cols_stop = 900 # None
if acquisition == 'ha_n140f005a500_gcv':
    row_of_interest = 100
    row_avg = 5
    cols_start = 700 # None
    cols_stop = 1100 # None
if acquisition == 'ha_n140f020a500_gcv':
    row_of_interest = 100
    row_avg = 5
    cols_start = 700 # None
    cols_stop = 1280 # None
if acquisition == 'ha_n140f010a500_gcv':
    row_of_interest = 100
    row_avg = 5
    cols_start = 700 # None
    cols_stop = 1280 # None
if acquisition == 'ha_n140f050a400_gcv':
    row_of_interest = 100
    row_avg = 5
    cols_start = 800 # None
    cols_stop = 1150 # None
if acquisition == 'ha_n140f100a300_gcv':
    row_of_interest = 130
    row_avg = 5
    cols_start = 250 # None
    cols_stop = 650 # None
if acquisition == 'ha_n140f200a400_gcv':
    row_of_interest = 80
    row_avg = 5
    cols_start = 300 # None
    cols_stop = 500 # None
if acquisition == 'ha_n140f500a1000_gcv':
    row_of_interest = 80
    row_avg = 5
    cols_start = 150 # None
    cols_stop = 350 # None


# <codecell>

plt.figure()
ax = plt.gca()

ax.imshow(frames[0])

ax.axhspan(row_of_interest-row_avg//2, row_of_interest+row_avg//2, color='k', alpha=.2)

if cols_start is not None:
    ax.axvline(cols_start, color='k', ls=':')
if cols_stop is not None:
    ax.axvline(cols_stop, color='k', ls=':')


# <codecell>



lumin = frames[:, row_of_interest-row_avg//2:row_of_interest+row_avg//2+1, cols_start:cols_stop].mean(axis=1)

threshold = 40 #todo otsu auto treshold here
if acquisition == 'ha_n140f200a400_gcv':
    threshold = 22.5 #todo otsu auto treshold here
if acquisition == 'ha_n140f500a1000_gcv':
    threshold = 14.5 #todo otsu auto treshold here


# <codecell>

fig, axes = plt.subplots(1, 2)

ax = axes[0]
ax.imshow(lumin, aspect='auto')

ax = axes[1]
ax.hist(lumin.flatten(), bins=np.arange(256+1))
ax.axvline(threshold, color='k', ls=':', lw=1)


# <codecell>

luminth = lumin > threshold

ix, iy = np.arange(luminth.shape[1]), np.arange(luminth.shape[0])

IX, IY = np.meshgrid(ix, iy)

#outer / inner | left / right

## left
#### outer
outl = np.argmin(luminth, axis=1)
### inner
luminth_findinl = luminth.copy()
luminth_findinl[IX < np.expand_dims(outl, 1)+1] = 0
innl = np.argmax(luminth_findinl, axis=1)

## right
### outer
outr = luminth.shape[1] - np.argmin(luminth[:, ::-1], axis=1)
### inner
luminth_findinr = luminth.copy()
luminth_findinr[IX > np.expand_dims(outr, 1)-1] = 0
innr = luminth.shape[1] - np.argmax(luminth_findinr[:, ::-1], axis=1)


# <codecell>

fig, axes = plt.subplots(1, 1, squeeze=False)

ax = axes[0, 0]
ax.imshow(luminth, aspect='auto', cmap = 'binary_r')
ax.plot(outl, np.arange(luminth.shape[0]))
ax.plot(innl, np.arange(luminth.shape[0]))
ax.plot(outr, np.arange(luminth.shape[0]))
ax.plot(innr, np.arange(luminth.shape[0]))


# <codecell>

meniscus_l_w = np.abs(innl - outl)
meniscus_r_w = np.abs(innr - outr)
center_w = np.abs(innl - innr)
menisci_w = meniscus_l_w + meniscus_r_w
full_w = menisci_w + center_w
centerpos = (innl + innr)/2
meniscpos = (outl + outr)/2


# <codecell>

10 * 500/707


# <codecell>

fig, axes = plt.subplots(2, 1, squeeze=False, sharex=True)

ax = axes[0, 0]
ax.plot(iy, meniscus_l_w, color='b', label='left menisc')
ax.plot(iy, meniscus_r_w, color='r', label='right menisc')
ax.plot(iy, center_w, color='orange', label='inner bridge width')
ax.plot(iy, menisci_w, color='green', label='menisci width')
ax.plot(iy, full_w, color='k', label='full bridge width')
ax.set_xlabel('time [frame]')
ax.set_ylabel('width [px]')
ax.legend()

ax = axes[1, 0]
ax.plot(iy, centerpos, color='b', label='centre')
ax.plot(iy, meniscpos, color='r', label='menisc')
ax.plot(iy, outl)
ax.plot(iy, outr)
# ax.plot(iy, center_w, color='k', label='bridge width')
ax.set_xlabel('time [frame]')
ax.set_ylabel('position [px]')
ax.legend()



# <codecell>

# # NO PADDING | It could be interesting to add a padding to smooth.
fft_window = 'hann'

f = utility.fourier.rdual(t)

m_pos_hat = utility.fourier.ft1d(meniscpos, fft_window)
c_pos_hat = utility.fourier.ft1d(centerpos, fft_window)

freq_m = utility.fourier.estimatesignalfrequency(meniscpos, t)
freq_c = utility.fourier.estimatesignalfrequency(centerpos, t)
utility.log_info(f'freqs: {freq_m} & {freq_c} Hz')
# freqmean = 


# <codecell>

# # OLD SCHOOL | With padding, before the invention of utility.fourier
# from scipy.signal import get_window
# 
# pad_size = 100
# fft_window = 'hann'
# 
# 
# f = np.fft.rfftfreq(len(t) + 2*pad_size, 1/acquisition_frequency)
# 
# m_pos_norm = meniscpos - meniscpos.mean()
# m_pos_norm /= m_pos_norm.max()
# m_pos_format = np.pad(m_pos_norm, pad_size) * get_window(fft_window, len(m_pos_norm) + 2*pad_size)
# m_pos_hat = np.fft.rfft(m_pos_format)
# 
# c_pos_norm = centerpos - centerpos.mean()
# c_pos_norm /= c_pos_norm.max()
# c_pos_format = np.pad(c_pos_norm, pad_size) * get_window(fft_window, len(c_pos_norm) + 2*pad_size)
# c_pos_hat = np.fft.rfft(c_pos_format)


# <codecell>



correll = m_pos_hat * np.conjugate(c_pos_hat)

maxfreq = f[np.argmax(np.abs(correll)**2)]

phase = np.angle(correll[np.argmax(np.abs(correll)**2)])
# if phase < np.pi:
#     phase += 2*np.pi
# print(f'phase: {round(phase*180/np.pi, 2)} deg')

print(f'Dataset {dataset} ; acquisition {acquisition}')
print('phase (deg)')
print(f'{round(phase*180/np.pi, 2)}')


# <codecell>

fig, axes = plt.subplots(2, 2, figsize=utility.figsize('double', None), squeeze=False, sharex='col')
ax = axes[0,0]
ax.set_title('Power spectrum')
ax.plot(f, np.abs(m_pos_hat)**2, c='k', label='menisc')
ax.plot(f, np.abs(c_pos_hat)**2, c='r', label='centre')
ax.plot(f, np.abs(correll)**2, c='g', label='cor')
ax.axvline(maxfreq, c='k', ls=':')
ax.legend()
ax.set_yscale('log')

ax = axes[0,1]
ax.set_title('Power spectrum')
ax.plot(f, np.abs(m_pos_hat)**2, c='k', label='menisc')
ax.plot(f, np.abs(c_pos_hat)**2, c='r', label='centre')
ax.plot(f, np.abs(correll)**2, c='g', label='cor')
ax.axvline(maxfreq, c='k', ls=':')
ax.legend()
ax.set_yscale('log')

ax = axes[1,0]
ax.set_title('Phase spectrum')
ax.plot(f, np.angle(m_pos_hat) - np.angle(c_pos_hat), c='b', label='menisc-centre')
ax.plot(f, np.angle(correll), c='g', label='cor')
ax.axvline(maxfreq, c='k', ls=':')
ax.legend()
ax.set_xlim(0, f.max())
ax.set_xlabel('Frequency (Hz)')

ax = axes[1,1]
ax.set_title('Phase spectrum')
ax.plot(f, np.angle(m_pos_hat) - np.angle(c_pos_hat), c='b', label='menisc-centre')
ax.plot(f, np.angle(correll), c='g', label='cor')
ax.axvline(maxfreq, c='k', ls=':')
ax.legend()
ax.set_xlim(maxfreq*0.90, maxfreq*1.1)
ax.set_xlabel('Frequency (Hz)')


# <codecell>



