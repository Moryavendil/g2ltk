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

px_per_mm = 707/5000


# <codecell>

15/px_per_mm


# <codecell>

# Acquisition selection
acquisitions = ['ha_n140f001a1000_gcv', 'ha_n140f001a500_gcv', 'ha_n140f002a500_gcv', 'ha_n140f002a800_gcv', 'ha_n140f005a500_gcv', 'ha_n140f010a500_gcv', 'ha_n140f020a500_gcv', 'ha_n140f050a400_gcv', 'ha_n140f100a300_gcv', 'ha_n140f200a400_gcv', 'ha_n140f500a1000_gcv']

freqexs = np.zeros(len(acquisitions), dtype=float)
phases = np.zeros(len(acquisitions), dtype=float)
ampls = np.zeros(len(acquisitions), dtype=float)

for i_acqu, acquisition in enumerate(acquisitions):
    acquisition_path = os.path.join(dataset_path, acquisition)
    datareading.is_this_a_video(acquisition_path)

    framenumbers = np.arange(datareading.get_number_of_available_frames(acquisition_path))[1:-1]
    roi = None, None, None, None  #start_x, start_y, end_x, end_y
    
    if acquisition == 'ha_n140f001a500_gcv':
        framenumbers = np.arange(182)
    
    length, height, width = datareading.get_geometry(acquisition_path, framenumbers=framenumbers, subregion=roi)
    # datareading.describe_acquisition(dataset, acquisition, framenumbers=framenumbers, subregion=roi)
    
    frames = datareading.get_frames(acquisition_path, framenumbers=framenumbers, subregion=roi)
    t = datareading.get_times(acquisition_path, framenumbers=framenumbers)
    acquisition_frequency = datareading.get_acquisition_frequency(acquisition_path)
    
    row_of_interest = height//2
    row_avg = 5
    cols_start = None
    cols_stop = None
    freqex = 0
    
    if acquisition == 'ha_n140f001a500_gcv':
        row_of_interest = 150
        row_avg = 5
        cols_start = 500  # None
        cols_stop = 900  # None
        freqex = 1.
    if acquisition == 'ha_n140f001a1000_gcv':
        row_of_interest = 150
        row_avg = 5
        cols_start = 500  # None
        cols_stop = 900  # None
        freqex = 1.
    if acquisition == 'ha_n140f002a500_gcv':
        row_of_interest = 150
        row_avg = 5
        cols_start = 500  # None
        cols_stop = 900  # None
        freqex = 2.
    if acquisition == 'ha_n140f002a800_gcv':
        row_of_interest = 150
        row_avg = 5
        cols_start = 500  # None
        cols_stop = 900  # None
        freqex = 2.
    if acquisition == 'ha_n140f005a500_gcv':
        row_of_interest = 100
        row_avg = 5
        cols_start = 700  # None
        cols_stop = 1100  # None
        freqex = 5.
    if acquisition == 'ha_n140f020a500_gcv':
        row_of_interest = 100
        row_avg = 5
        cols_start = 700  # None
        cols_stop = 1280  # None
        freqex = 20.
    if acquisition == 'ha_n140f010a500_gcv':
        row_of_interest = 100
        row_avg = 5
        cols_start = 700  # None
        cols_stop = 1280  # None
        freqex = 10.
    if acquisition == 'ha_n140f050a400_gcv':
        row_of_interest = 100
        row_avg = 5
        cols_start = 800  # None
        cols_stop = 1150  # None
        freqex = 50.
    if acquisition == 'ha_n140f100a300_gcv':
        row_of_interest = 130
        row_avg = 5
        cols_start = 250  # None
        cols_stop = 650  # None
        freqex = 100.
    if acquisition == 'ha_n140f200a400_gcv':
        row_of_interest = 80
        row_avg = 5
        cols_start = 300  # None
        cols_stop = 500  # None
        freqex = 200.
    if acquisition == 'ha_n140f500a1000_gcv':
        row_of_interest = 80
        row_avg = 5
        cols_start = 150  # None
        cols_stop = 350  # None
        freqex = 500.
    
    lumin = frames[:, row_of_interest - row_avg // 2:row_of_interest + row_avg // 2 + 1, cols_start:cols_stop].mean(axis=1)
    
    threshold = 40  #todo otsu auto treshold here
    if acquisition == 'ha_n140f200a400_gcv':
        threshold = 22.5  #todo otsu auto treshold here
    if acquisition == 'ha_n140f500a1000_gcv':
       threshold = 14.5  #todo otsu auto treshold here
    luminth = lumin > threshold
    
    ix, iy = np.arange(luminth.shape[1]), np.arange(luminth.shape[0])
    
    IX, IY = np.meshgrid(ix, iy)
    
    #outer / inner | left / right
    
    ## left
    #### outer
    outl = np.argmin(luminth, axis=1)
    ### inner
    luminth_findinl = luminth.copy()
    luminth_findinl[IX < np.expand_dims(outl, 1) + 1] = 0
    innl = np.argmax(luminth_findinl, axis=1)
    
    ## right
    ### outer
    outr = luminth.shape[1] - np.argmin(luminth[:, ::-1], axis=1)
    ### inner
    luminth_findinr = luminth.copy()
    luminth_findinr[IX > np.expand_dims(outr, 1) - 1] = 0
    innr = luminth.shape[1] - np.argmax(luminth_findinr[:, ::-1], axis=1)

    meniscus_l_w = np.abs(innl - outl)
    meniscus_r_w = np.abs(innr - outr)
    center_w = np.abs(innl - innr)
    menisci_w = meniscus_l_w + meniscus_r_w
    full_w = menisci_w + center_w
    centerpos = (innl + innr) / 2
    meniscpos = (outl + outr) / 2

    from scipy.signal import get_window
    
    pad_size = 0
    fft_window = 'hann'
    
    f = np.fft.rfftfreq(len(t) + 2 * pad_size, 1 / acquisition_frequency)
    
    m_pos_norm = meniscpos - meniscpos.mean()
    m_pos_norm /= m_pos_norm.max()
    m_pos_format = np.pad(m_pos_norm, pad_size) * get_window(fft_window, len(m_pos_norm) + 2 * pad_size)
    m_pos_hat = np.fft.rfft(m_pos_format)
    
    c_pos_norm = centerpos - centerpos.mean()
    c_pos_norm /= c_pos_norm.max()
    c_pos_format = np.pad(c_pos_norm, pad_size) * get_window(fft_window, len(c_pos_norm) + 2 * pad_size)
    c_pos_hat = np.fft.rfft(c_pos_format)
    
    correll = m_pos_hat * np.conjugate(c_pos_hat)
    
    maxfreq = f[np.argmax(np.abs(correll) ** 2)]
    
    phase = np.angle(correll[np.argmax(np.abs(correll) ** 2)])
    # if phase < np.pi:
    #     phase += 2 * np.pi
    # print(f'phase: {round(phase*180/np.pi, 2)} deg')
    
    print(f'Dataset {dataset} ; acquisition {acquisition} \t| phase = {round(phase * 180 / np.pi, 2)} deg')

    freqexs[i_acqu] = freqex
    phases[i_acqu] = phase
    ampls[i_acqu] = centerpos.max() - centerpos.min()
    ampls[i_acqu] = (np.sqrt(utility.fourier.power_near_peak1d(freqex, utility.fourier.psd1d(centerpos, f), peak_depth_dB=20, x=f)))



# <codecell>

plt.figure()
ax = plt.gca()

ax.scatter(freqexs * ampls, phases*180/np.pi)
ax.set_ylim(-180, 0)
ax.set_ylabel('phase')
ax.set_xlabel('frequency * amplitude')


# <codecell>

10 / px_per_mm


# <codecell>

70*300*6/1000000


# <codecell>

plt.figure()
ax = plt.gca()

ax.scatter(freqexs, phases*180/np.pi)
ax.set_ylim(-180, 0)
ax.set_ylabel('phase')
ax.set_xlabel('frequency')


# <codecell>

plt.figure()
ax = plt.gca()

ax.scatter(freqexs, ampls)
ax.set_ylabel('amplitude')
ax.set_xlabel('frequency')


# <codecell>

k

