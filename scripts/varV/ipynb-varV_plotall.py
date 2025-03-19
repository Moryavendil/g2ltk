# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>
# <codecell>

# %matplotlib notebook

import os
import numpy as np
import matplotlib.pyplot as plt

from g2ltk import set_verbose, datareading, datasaving, utility
utility.configure_mpl()


# <codecell>

### Datasets display
root_path = '../'
datasets = datareading.find_available_datasets(root_path)
print('Available datasets:', datareading.find_available_datasets(root_path))


# <codecell>

### Dataset selection & acquisitions display
dataset = '-'
if len(datasets) == 1:
    dataset = datasets[0]
    datareading.log_info(f'Auto-selected dataset {dataset}')
dataset_path = os.path.join(root_path, dataset)
datareading.describe_dataset(dataset_path, type='gcv', makeitshort=True)


# <codecell>

### meta-info reading
import pandas as pd

workbookpath = os.path.join(dataset_path, 'datasheet.xlsx')

metainfo = pd.read_excel(workbookpath, sheet_name='metainfo', skiprows=2)

metakeys = [key for key in metainfo.keys() if 'unnamed' not in key.lower()]
metavalid = metainfo['acquisition_title'].astype(str) != 'nan'

m_acquisition_title = metainfo['acquisition_title'][metavalid].to_numpy()
m_excitation_amplitude        = metainfo['excitation_amplitude'][metavalid].to_numpy()


# <codecell>


data_sheetname = 'autodata'

df_autodata = pd.read_excel(workbookpath, sheet_name=data_sheetname)
datakeys = [key for key in df_autodata.keys() if 'unnamed' not in key.lower()]

datavalid = df_autodata['f0'].astype(str) != 'nan'
instab = df_autodata['q0'].astype(str) != 'nan'

N_acq = np.sum(datavalid)

acquisition_title = df_autodata['acquisition_title'][datavalid].to_numpy()
f0 = df_autodata['f0'][datavalid].to_numpy()
q0 = df_autodata['q0'][datavalid].to_numpy()
fdrift = df_autodata['fdrift'][datavalid].to_numpy()
z0 = df_autodata['z0'][datavalid].to_numpy()
z1 = df_autodata['z1'][datavalid].to_numpy()
w1 = df_autodata['w1'][datavalid].to_numpy()


# <codecell>

plt.figure()
ax = plt.gca()
ax.scatter(m_excitation_amplitude, z0)
ax.set_xlabel('Driving signal amplitude [mV]')
ax.set_ylabel('$z_0$ [px]')


# <codecell>

plt.figure()
ax = plt.gca()
ax.scatter(z0[instab], q0[instab])
# 
# ax.scatter(z0[instab], 0.0027*np.sqrt(1 + (2*np.pi*q0[instab] * z1[instab])**2), color='r')
ax.scatter(z0[instab], q0[instab] / (1 + (2*np.pi*q0[instab]*z1[instab])**2/4), color='r', label='q (deformation corrected)')

ax.set_xlabel('$z_0$ [px]')
ax.set_ylabel('spatial frequency $q$ [px-1]')
ax.set_ylim(0, ax.get_ylim()[1])
ax.legend()


# <codecell>

plt.figure()
ax = plt.gca()
ax.scatter(z0[instab], fdrift[instab])
ax.set_xlabel('$z_0$ [px]')
ax.set_ylabel('drift frequency $fdrift$ [frame-1]')
# ax.set_ylim(0, ax.get_ylim()[1])


# <codecell>

plt.figure()
ax = plt.gca()
ax.scatter(z0, z1, label='z1')
ax.scatter(z0, w1, label='w1')
ax.plot(z0, z0, label='z0', color='gray', ls='--')
ax.set_xlabel('$z_0$ [px]')
ax.set_ylabel('Amplitude [px]')
ax.legend()


# <codecell>

z0_thresh_alamano = 13.
z1_factor_alamano = 18.
w1_factor_alamano = z1_factor_alamano / 2.5# 7.

z0_test = np.linspace(z0.min(), z0.max(), 1000)
w1_alamano = np.zeros_like(z0_test)
w1_alamano[z0_test > z0_thresh_alamano] = w1_factor_alamano*np.sqrt(z0_test[z0_test > z0_thresh_alamano] - z0_thresh_alamano)
z1_alamano = np.zeros_like(z0_test)
z1_alamano[z0_test > z0_thresh_alamano] = z1_factor_alamano*np.sqrt(z0_test[z0_test > z0_thresh_alamano] - z0_thresh_alamano)

plt.figure()
ax = plt.gca()
ax.scatter(z0, z1, label='z1')
ax.scatter(z0, w1, label='w1')
ax.plot(z0, z0, label='z0', color='gray', ls='--')
ax.plot(z0_test, z1_alamano, label='z1 alamano')
ax.plot(z0_test, w1_alamano, label='w1 alamano')
ax.set_xlabel('$z_0$ [px]')
ax.set_ylabel('Amplitude [px]')
ax.legend()


# <codecell>

plt.figure()
ax = plt.gca()
ax.scatter(z0[instab], z1[instab]/w1[instab], label='z1')

# ax.plot(z0, z0, label='z0', color='gray', ls='--')
# ax.set_xlabel('$z_0$ [px]')
# ax.set_ylabel('Amplitude [px]')
# ax.legend()


# <codecell>



