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
dataset = '40evo'

datareading.describe_dataset(dataset=dataset, videotype='gcv', makeitshort=True)
dataset_path = datareading.generate_dataset_path(dataset)


# <codecell>

### meta-info reading
import pandas as pd
import os

workbookpath = os.path.join(dataset_path, 'datasheet.xlsx')

metainfo = pd.read_excel(workbookpath, sheet_name='metainfo', skiprows=2)

metakeys = [key for key in metainfo.keys() if 'unnamed' not in key.lower()]
metavalid = metainfo['acquisition_title'].astype(str) != 'nan'

m_acquisition_title = metainfo['acquisition_title'][metavalid].to_numpy()
m_excitation_amplitude        = metainfo['excitation_amplitude'][metavalid].to_numpy()


# <codecell>

px_per_mm = (1563/30+1507/30)/2
px_per_um = px_per_mm * 1e3
fr_per_s = (40 * 100)
fr_per_ms = fr_per_s / 1000

mm_per_px = 1/px_per_mm

data_sheetname = 'autodata'

df_autodata = pd.read_excel(workbookpath, sheet_name=data_sheetname)
datakeys = [key for key in df_autodata.keys() if 'unnamed' not in key.lower()]

datavalid = df_autodata['f0'].astype(str) != 'nan'
instab = df_autodata['q0'].astype(str) != 'nan'

N_acq = np.sum(datavalid)

acquisition_title = df_autodata['acquisition_title'][datavalid].to_numpy()
f0 = df_autodata['f0'][datavalid].to_numpy() * fr_per_s
q0 = df_autodata['q0'][datavalid].to_numpy() * px_per_mm
fdrift = df_autodata['fdrift'][datavalid].to_numpy() * fr_per_s
z0 = df_autodata['z0'][datavalid].to_numpy() / px_per_mm 
z1 = df_autodata['z1'][datavalid].to_numpy() / px_per_mm 
w1 = df_autodata['w1'][datavalid].to_numpy() / px_per_mm

k0 = 2*np.pi*q0
omega0 = 2*np.pi*f0


# <codecell>

slope = 2/1000
seuil = 48
def z0_th(V, seuil, slope):
    return (V - seuil) * slope

V_max_for_fit = 195
crit = m_excitation_amplitude < V_max_for_fit
from scipy.optimize import curve_fit
popt, pcov = curve_fit(z0_th, m_excitation_amplitude[crit], z0[crit], p0=(seuil, slope))

fig, axes = plt.subplots(1,1, squeeze=False, sharex=True)
ax = axes[0,0]
ax.scatter(m_excitation_amplitude, z0, label='exp pts')
# ax.plot(Vrpz, z0_th(Vrpz, *popt))
ax.set_xlabel('Driving signal amplitude [mV]')
ax.set_ylabel('$z_0$ [mm]')


# ax = axes[1,0]
# ax.scatter(m_excitation_amplitude, z0 - z0_th(m_excitation_amplitude, *popt), label='exp pts')
# ax.set_xlabel('Driving signal amplitude [mV]')
# ax.set_ylabel('$z_0$ [mm]')


# <codecell>

### ATTENUATION
b = .6
nu = 1.
mu = 1 / (b ** 2 / (12 * nu))
print(f'mu = {round(mu, 3)} Hz')

### BULK SPEED
g = 9.81e3
u0 = g / mu
print(f'u0 = {round(u0, 3)} mm/s')

### CAPILLARY SPEED
gamma = 17
rho = 1.72e-3
w0 = 0.17723893195382692
w0_u = np.sqrt(0.001713682346235076**2 + (1/px_per_mm/5)**2) # combine the standard deviation and the pixel size
# w0_u = np.sqrt(w0_u**2 + (1*mm_per_px)**2) # we might be overestimating w of something like 1 px because of bad lighting condition
print(f'w0 = {round(w0, 3)} pm {round(w0_u, 3)} mm')


Gamma = np.pi * gamma / (2 * rho)
vc = np.sqrt(Gamma / w0)
vc_u = vc * max(0.01, np.sqrt((1/2 * w0_u/w0)**2))
print(f'vc = {round(vc, 3)} pm {round(vc_u, 3)} mm/s')

fvrai = 40
omegavrai = 2*np.pi*fvrai

qth = fvrai / u0


# <codecell>


z0test = np.linspace(0, .5, 1000)[1:-1]
z0threshold = 0.252

color_q = '#9c1ab2'
color_w = '#3d5da9'
color_z = '#ff1a1a' # (dark version : #9c1ab2 | light version : #c320df )

incert_q = np.zeros_like(z0)
incert_z1 = np.zeros_like(z0)
incert_w1 = np.zeros_like(z0)

incert_q = 1/10 * q0 / 4
incert_z1 = np.full_like(z0, mm_per_px / 4)
incert_w1 = np.full_like(z0, mm_per_px / 4)
incert_z1[instab] += z1[instab] * .01
incert_w1[instab] += w1[instab] * .01

A = 0.9


# <codecell>

# utility.activate_saveplot()


# <codecell>

fig, ax = plt.subplots(figsize=utility.figsize('wide', ratio = 1.6))
errorbar_kw = {'capsize': 3, 'ls': '', 'marker':'o', 'mfc': 'w'}

ax.errorbar(z0[instab], q0[instab], yerr=incert_q[instab], color=color_q, label='$q = k/(2\pi)$', **errorbar_kw)
# ax.plot(z0[instab], qth * (1 + (z1[instab]/A)**2))

ax.axhline(qth, ls='--', label='$q_{model} = f_0 / u_0$', alpha=.6, color=color_q)
# ax.axvline(z0threshold, ls=':', label='Threshold', color='g')

ax.set_xlim(.1, .45)
ax.set_xlabel('Transverse movement amplitude $|Z_0|$ [mm]')

ax.set_ylim(0, .25)
ax.set_ylabel('Spatial frequency [mm$^{-1}$]')
ax.legend()

# utility.save_graphe('saturation_wavelengthchange')


# <codecell>


plt.figure()
ax = plt.gca()
ax.scatter(z0[instab], fdrift[instab])
ax.set_xlabel('$z_0$ [mm]')
ax.set_ylabel('drift frequency $fdrift$ [Hz]')
# ax.set_ylim(0, ax.get_ylim()[1])


# <codecell>

fig, ax = plt.subplots(figsize=utility.figsize('wide', ratio = 1.4))
errorbar_kw = {'capsize': 3, 'ls': '', 'marker':'o', 'mfc': 'w'}

ax.errorbar(z0, z1, yerr=incert_z1, color=color_z, label=r'$|Z_1|$', **errorbar_kw)
ax.errorbar(z0, w1, yerr=incert_w1, color=color_w, label=r'$|w_1|$', **errorbar_kw)
ax.plot(z0test, z0test, label='$|Z_0|$', color='k', ls='--', alpha=.6)

ax.set_xticks(np.arange(0, .6, .1))
ax.set_xlim(0., .45)
ax.set_xlabel('Transverse movement amplitude $|Z_0|$ [mm]')

ax.set_yticks(np.arange(0, .6, .1))
ax.set_ylim(0, .45)
ax.set_ylabel('Amplitude [mm]')
ax.legend()

ax.set_aspect('equal')

# plt.tight_layout()
# utility.save_graphe('saturation_amplitudes')


# <codecell>

plt.close()


# <codecell>

z0_thresh_alamano = .25
z1_factor_alamano = 2
w1_factor_alamano = 2.5# 7.

z0_test = np.linspace(z0.min(), z0.max(), 1000)

z1_alamano = np.zeros_like(z0_test)
inside_sqrt = z0_test - z0_thresh_alamano
z1_alamano[inside_sqrt > 0] = z1_factor_alamano*np.sqrt(inside_sqrt[inside_sqrt > 0])

w1_alamano = z1_alamano / w1_factor_alamano

plt.figure()
ax = plt.gca()
ax.scatter(z0, z1, label='z1')
ax.scatter(z0, w1, label='w1')
ax.plot(z0, z0, label='z0', color='gray', ls='--')
ax.plot(z0_test, z1_alamano, label='z1 alamano')
ax.plot(z0_test, w1_alamano, label='w1 alamano')
ax.set_xlabel('$z_0$ [mm]')
ax.set_ylabel('Amplitude [mm]')
ax.legend()


# <codecell>




# <codecell>

# plt.figure()
# ax = plt.gca()
# ax.scatter(z0[instab], w1[instab]/z1[instab], label='z1')
# 
# z0test = np.linspace(0, 0.5)
# 
# eta = 0
# 
# F = (2*np.pi* f0)**3 * z0**2 / (mu**2 * w0 * vc)
# 
# 
# print(f'F={F}')
# lam = mu/2 * (1-eta)/2 * (-1 + np.sqrt(1 + (eta + F)/((1-eta)/2)**2))
# fact = (2*np.pi* f0) * (2*np.pi* q0) / (mu + 2*lam)
# 
# ax.plot(z0[instab], z0[instab] * fact[instab], label='z0', color='gray', ls='--')
# ax.set_xlabel('$z_0$ [mm]')
# ax.set_ylabel('W / Z0')
# ax.set_ylim(0, 1)
# # ax.legend()


# <codecell>


eta = -0

muf = mu
muw = mu

F = (2*np.pi* fvrai)**3 * (z0test/2)**2 / (muw*muf * w0 * vc)

z0seuil = z0test[np.argmin((F + eta)**2)]

Delta = - (muf/2)*(muw/2) * (eta + F) # matrix determinant
tr = -muw/2 + eta * muf/2 # matrix trace

D = tr**2 - 4*Delta # system determinant
lam = (tr + np.sqrt(D))/2 # biggest eigenvalue

den = muw + 2*lam
den[den==0] = 1e-10
fact = (2*np.pi* fvrai) / den

ratio = np.sqrt(w0 * vc / (2*np.pi*fvrai) * (-eta*mu + 2*lam) / (mu + 2*lam))

fig, axes = plt.subplots(3, sharex=True, figsize=utility.figsize('wide', ratio=0.7))

ax = axes[0]
ax.scatter(z0[instab], w1[instab]/z1[instab] / (2*np.pi*q0[instab]), label='z1')
ax.plot(z0test, np.ones_like(z0test)*ratio, label='z0', color='gray', ls='--')
ax.axvline(z0seuil, color='k', lw=.5)
ax.set_ylim(0, 1)

ax = axes[1]
ax.plot(z0test, F, label='F', color='r', ls='-')
ax.axhline(-eta, color='r', lw=.5, alpha=.8)
ax.axvline(z0seuil, color='k', lw=.5)

ax.legend()

# ax = axes[2]
# ax.plot(z0test,Delta, color='r', ls='-')
# ax.axhline(0, color='r', lw=.5, alpha=.8)
# ax.legend()
# 
# ax = axes[3]
# ax.plot(z0test,np.sqrt(D), color='r', ls='-')
# ax.axhline(-tr, color='r', lw=.5, alpha=.8)
# ax.legend()

ax = axes[-1]
ax.plot(z0test, lam, label=r'$\lambda_+$', color='g', ls='--')
# ax.plot(z0test, lam, label=r'$\lambda_+$', color='g', ls='--')
ax.axhline(0, color='g', lw=.5, alpha=.8)
ax.axvline(z0seuil, color='k', lw=.5)
ax.legend()


ax.set_xlabel('$z_0$ [mm]')
# ax.set_ylabel('W / Z0 / k')
# ax.legend()


# <codecell>

crit = instab & (z0 > 0.253)

ratio_exp = w1 / z1 / k0
ratio_exp_u = ratio_exp * np.sqrt((incert_z1 / z1) ** 2 + (incert_w1 / w1) ** 2 + (incert_q / q0) ** 2)

ratio_th = np.sqrt( w0 * vc / omegavrai )
ratio_th_u = ratio_th * np.sqrt((1/2 * w0_u / w0) ** 2 + (1/2 * vc_u / vc) ** 2)


### WITH THE LAMBDA
eta = 0

F = (omegavrai)**3 * (z0test/2)**2 / (mu*mu * w0 * vc) # order parameter

def lam(eta_):
    Delta = - (mu/2)*(mu/2) * (eta_ + F) # matrix determinant
    tr = -mu/2 + eta_ * mu/2 # matrix trace
    D = tr**2 - 4*Delta # system determinant
    return (tr + np.sqrt(D))/2 # biggest eigenvalue

ratio_fin_min = ratio_th * np.sqrt((-(0)*mu + 2*lam(0)) / (mu + 2*lam(0))) # eta = 0
ratio_fin_max = ratio_th * np.sqrt((-(-1)*mu + 2*lam(-1)) / (mu + 2*lam(-1))) # eta = -1
ratio_fin = (ratio_fin_max + ratio_fin_min)/2
ratio_fin_u = (ratio_fin_max - ratio_fin_min)/2

fig, ax = plt.subplots(figsize=utility.figsize('wide', ratio = 1.6))
errorbar_kw = {'capsize': 3, 'ls': '', 'marker':'o', 'mfc': 'w'}

ax.errorbar(z0[crit], ratio_exp[crit], yerr=ratio_exp_u[crit], color=color_q, label='Measurements', **errorbar_kw)

# ax.axhline(ratio_th, label=r'$\sqrt{w_0\, v_c / \omega_0}$', color=color_q, ls='-', alpha=.6)
# ax.axhspan(ratio_th - ratio_th_u/2, ratio_th + ratio_th_u/2, color=color_q, ls='', alpha=.1)

ax.plot(z0test, ratio_fin, color=color_q, ls='-', alpha=.6, label='From linear theory')
ax.fill_between(z0test, ratio_fin_max, ratio_fin_min, color=color_q, alpha=.1, lw=0.0, label='From linear theory')

ax.axvline(0.252, label='Threshold', color='k', ls=':', alpha=.6)

ax.set_xticks(np.arange(0, .6, .05))
ax.set_xlim(0.2, .45)
ax.set_xlabel('Transverse movement amplitude $|Z_0|$ [mm]')

ax.set_yticks(np.arange(0, 1, .2))
ax.set_ylim(0, .8)
ax.set_ylabel('$|W_1| / (k\,|Z_1|)$ [mm]')

ax.legend()

# plt.tight_layout()

# utility.save_graphe('saturation_ratio_amplitudes')


# <codecell>

plt.close()


# <codecell>




# <codecell>




# <codecell>




# <codecell>




# <codecell>




# <codecell>




# <codecell>




# <codecell>




# <codecell>




# <codecell>




# <codecell>



