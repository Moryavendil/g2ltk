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
f0_exp = df_autodata['f0'][datavalid].to_numpy() * fr_per_s
q0_exp = df_autodata['q0'][datavalid].to_numpy() * px_per_mm
fdrift = df_autodata['fdrift'][datavalid].to_numpy() * fr_per_s
F_exp = df_autodata['z0'][datavalid].to_numpy() / px_per_mm 
Z_exp = df_autodata['z1'][datavalid].to_numpy() / px_per_mm 
W_exp = df_autodata['w1'][datavalid].to_numpy() / px_per_mm

k0_exp = 2 * np.pi * q0_exp
omega0_exp = 2 * np.pi * f0_exp


color_q = '#9c1ab2'
color_w = '#3d5da9'
color_z = '#ff1a1a' # (dark version : #9c1ab2 | light version : #c320df )

incert_q = np.zeros_like(F_exp)
incert_z1 = np.zeros_like(F_exp)
incert_w1 = np.zeros_like(F_exp)

incert_q = 1 / 10 * q0_exp / 4
incert_z1 = np.full_like(F_exp, mm_per_px / 4)
incert_w1 = np.full_like(F_exp, mm_per_px / 4)
incert_z1[instab] += Z_exp[instab] * .01
incert_w1[instab] += W_exp[instab] * .01

ratio_exp = W_exp / Z_exp / k0_exp
ratio_exp_u = ratio_exp * np.sqrt((incert_z1 / Z_exp) ** 2 + (incert_w1 / W_exp) ** 2 + (incert_q / q0_exp) ** 2)


# <codecell>

slope = 2/1000
seuil = 48
def z0_th(V, seuil, slope):
    return (V - seuil) * slope

V_max_for_fit = 195
crit = m_excitation_amplitude < V_max_for_fit
from scipy.optimize import curve_fit
popt, pcov = curve_fit(z0_th, m_excitation_amplitude[crit], F_exp[crit], p0=(seuil, slope))

fig, axes = plt.subplots(1,1, figsize=utility.figsize('simple'), squeeze=False, sharex=True)
ax = axes[0,0]
ax.scatter(m_excitation_amplitude, F_exp, label='exp pts')
# ax.plot(Vrpz, z0_th(Vrpz, *popt))
ax.set_xlabel('Driving signal amplitude [mV]')
ax.set_ylabel('$z_0$ [mm]')


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

f0 = 40
omega0 = 2 * np.pi * f0



# <codecell>


Z_0 = np.sqrt(mu**2 * w0 * vc / omega0**3)

# VALEUR NAIVE
omegaz_v0 = 0
omegaw_v0 = omegaz_v0 + omega0
k_v0 = omegaw_v0 / u0 

fz_v0 = omegaz_v0 / (2*np.pi)
fw_v0 = omegaw_v0 / (2*np.pi)
q_v0 = k_v0 / (2*np.pi)

phi_v0 = np.sqrt(omega0 * vc / w0) / (u0 * k_v0)

Fc_0 = 0.25

# rato = W / (kZ) = k Fc / phi
ETA = .8 # 0.8 WORKS BEST
ratio_v0 = np.sqrt(ETA) * np.sqrt(u0 ** 2 * w0 / (omega0 * vc))



# <codecell>

fig, axes = plt.subplots(4, 1, figsize=utility.figsize('wide', ratio = .8), sharex=True, squeeze=False)
errorbar_kw = {'capsize': 3, 'ls': '', 'marker':'o', 'mfc': 'w'}


ax = axes[0, 0]
ax.set_title(r'$\omega_z(F)$')
ax.scatter(F_exp[instab], fdrift[instab], color=color_z, label=r'$f_z$ (exp.)')
ax.axhline(0, ls='--', label='$\omega_z = 0$', alpha=.6, color=color_z)

ax.set_ylim(-1, 1)
ax.set_ylabel('$\omega_z/(2\pi)$ [Hz]')
ax.legend()


ax = axes[1, 0]
ax.set_title('q(F)')
ax.errorbar(F_exp[instab], q0_exp[instab], yerr=incert_q[instab], color=color_q, label=r'$q$ (exp.)', **errorbar_kw)
ax.axhline(q_v0, ls='--', label='$\omega_w/ v_w = f_0 / u_0$', alpha=.6, color=color_q)

ax.set_ylim(0, .25)
ax.set_ylabel('$k/(2\pi)$ [mm$^{-1}$]')
ax.legend()


ax = axes[2,0]
ax.scatter(F_exp, Z_exp, label=r'$|{Z}|$ (exp.)', color=color_z)
ax.scatter(F_exp, W_exp, label=r'$|{W}|$ (exp.)', color=color_w)
ax.plot(F_exp, F_exp, label=r'$|{F}|$', color='k', alpha=.3, ls=':')

ax.set_ylabel('Amplitude [mm]')
ax.legend()


ax = axes[3,0]
ax.errorbar(F_exp[instab], ratio_exp[instab], yerr=ratio_exp_u[instab], color=color_q, label='Measurements', **errorbar_kw)
ax.axhline(ratio_v0, ls='--', label='ratui', alpha=.6, color=color_q)

ax.set_ylim(0, 1)
ax.set_ylabel(r'Amplitude ratio'+'\n'+'$|W| / (k\,|Z|)$ [mm]')

axes[-1, 0].set_xlim(.1, .42)
axes[-1, 0].set_xlabel('Transverse movement amplitude $|F|$ [mm]')


# <codecell>

fig, axes = plt.subplots(3, 1, figsize=utility.figsize('wide', ratio = .8), sharex=True, squeeze=False)
errorbar_kw = {'capsize': 3, 'ls': '', 'marker':'o', 'mfc': 'w'}


ax = axes[0, 0]
ax.set_title('q(F)')
ax.errorbar(F_exp[instab], (q0_exp[instab]-q_v0)**2, color=color_q, label=r'$q$ (exp.)', **errorbar_kw)

# ax.set_ylim(0, .25)
# ax.set_ylabel('$k/(2\pi)$ [mm$^{-1}$]')
ax.legend()


ax = axes[1,0]
ax.scatter(F_exp, Z_exp**2, label=r'$|{Z}|$ (exp.)', color=color_z)
ax.scatter(F_exp, W_exp**2, label=r'$|{W}|$ (exp.)', color=color_w)

# ax.set_ylabel('Amplitude [mm]')
ax.legend()


ax = axes[2,0]
ax.scatter(F_exp, Z_exp**4, label=r'$|{Z}|$ (exp.)', color=color_z)
ax.scatter(F_exp, W_exp**4, label=r'$|{W}|$ (exp.)', color=color_w)

# ax.set_ylabel('Amplitude [mm]')
ax.legend()


axes[-1, 0].set_xlim(.1, .42)
axes[-1, 0].set_xlabel('Transverse movement amplitude $|F|$ [mm]')


# <codecell>

Ftest = np.linspace(0.1, 0.45, 500)

# NOW THE FIT


def Z_fit(F, Fc, A, B):
    return B*np.maximum(F - Fc, 0)**(1/4)

def q_fit(F, Fc, A, B):
    return q_v0 + A*Z_fit(F, Fc, A, B)**2

def W_fit(F, Fc, A, B):
    return ratio_v0 * Z_fit(F, Fc, A, B) * 2*np.pi*q_fit(F, Fc, A, B)

Fc_guess = 0.25
A_guess = .3
B_guess = .6

p_guess = [Fc_guess, A_guess, B_guess]


# <codecell>

valid_for_fit = F_exp > 0.265
valid_for_fit = F_exp > 0.18

def minfn(FcAB):
    DeltaZ = Z_exp[valid_for_fit] - Z_fit(F_exp[valid_for_fit], *FcAB)
    Z_ref = Z_0
    DeltaW = W_exp[valid_for_fit] - W_fit(F_exp[valid_for_fit], *FcAB)
    W_ref = ratio_v0 * Z_ref * 2*np.pi*q_v0
    Deltaq = q0_exp[valid_for_fit] - q_fit(F_exp[valid_for_fit], *FcAB)
    q_ref = q_v0
    return np.sum( (DeltaZ / Z_ref)**2 + (DeltaW / W_ref)**2 + (Deltaq / q_ref)**2 )

def minfn_fixedthreshold(AB):
    FcAB = [Fc_guess, *AB]
    DeltaZ = Z_exp[valid_for_fit] - Z_fit(F_exp[valid_for_fit], *FcAB)
    Z_ref = Z_0
    DeltaW = W_exp[valid_for_fit] - W_fit(F_exp[valid_for_fit], *FcAB)
    W_ref = ratio_v0 * Z_ref * 2*np.pi*q_v0
    Deltaq = q0_exp[valid_for_fit] - q_fit(F_exp[valid_for_fit], *FcAB)
    q_ref = q_v0
    return np.sum( (DeltaZ / Z_ref)**2 + (DeltaW / W_ref)**2 + (Deltaq / q_ref)**2 )

from scipy.optimize import minimize

FcAB = minimize(minfn, p_guess).x
# FcAB = [Fc_guess, *minimize(minfn_fixedthreshold, p_guess[1:]).x]


# <codecell>

fig, axes = plt.subplots(4, 1, figsize=utility.figsize('wide', ratio = .8), sharex=True, squeeze=False)
errorbar_kw = {'capsize': 3, 'ls': '', 'marker':'o', 'mfc': 'w'}


ax = axes[0, 0]
ax.set_title(r'$\omega_z(F)$')
ax.scatter(F_exp[instab], fdrift[instab], color=color_z, label=r'$f_z$ (exp.)')
ax.axhline(0, ls='--', label='$\omega_z = 0$', alpha=.6, color=color_z)
# ax.plot(Ftest, 0*np.ones_like(Ftest), ls='-', label='fit', alpha=.8, color=color_z)

ax.set_ylim(-1, 1)
ax.set_ylabel('$\omega_z/(2\pi)$ [Hz]')
ax.legend()


ax = axes[1, 0]
ax.set_title('q(F)')
ax.errorbar(F_exp[instab], q0_exp[instab], yerr=incert_q[instab], color=color_q, label=r'$q$ (exp.)', **errorbar_kw)
ax.axhline(q_v0, ls='--', label='$\omega_w/ v_w = f_0 / u_0$', alpha=.6, color=color_q)
# ax.plot(Ftest, q_fit(Ftest, *p_guess), ls='-', label='fit', alpha=.8, color=color_q)
ax.plot(Ftest, q_fit(Ftest, *FcAB), ls='-', label='fit', alpha=.8, color=color_q)

ax.set_ylim(0, .25)
ax.set_ylabel('$k/(2\pi)$ [mm$^{-1}$]')
ax.legend()


ax = axes[2,0]
ax.scatter(F_exp, Z_exp, label=r'$|{Z}|$ (exp.)', color=color_z)
ax.scatter(F_exp, W_exp, label=r'$|{W}|$ (exp.)', color=color_w)
# ax.plot(Ftest, Z_fit(Ftest, *p_guess), ls='-', label='fit', alpha=.8, color=color_z)
# ax.plot(Ftest, W_fit(Ftest, *p_guess), ls='-', label='fit', alpha=.8, color=color_w)
ax.plot(Ftest, Z_fit(Ftest, *FcAB), ls='-', label='fit', alpha=.8, color=color_z)
ax.plot(Ftest, W_fit(Ftest, *FcAB), ls='-', label='fit', alpha=.8, color=color_w)
ax.plot(F_exp, F_exp, label=r'$|{F}|$', color='k', alpha=.3, ls=':')

ax.set_ylabel('Amplitude [mm]')
ax.legend()


ax = axes[3,0]
ax.errorbar(F_exp[instab], ratio_exp[instab], yerr=ratio_exp_u[instab], color=color_q, label='Measurements', **errorbar_kw)
ax.axhline(ratio_v0, ls='--', label='ratui', alpha=.6, color=color_q)

ax.set_ylim(0, 1)
ax.set_ylabel(r'Amplitude ratio'+'\n'+'$|W| / (k\,|Z|)$ [mm]')

axes[-1, 0].set_xlim(.1, .42)
axes[-1, 0].set_xlabel('Transverse movement amplitude $|F|$ [mm]')


# <codecell>

z0_thresh_alamano = .22
z1_factor_alamano = .7
w1_factor_alamano = 2.5# 7.

z0_test = np.linspace(F_exp.min(), F_exp.max(), 1000)

z1_alamano = np.zeros_like(z0_test)
inside_sqrt = z0_test - z0_thresh_alamano
z1_alamano[inside_sqrt > 0] = z1_factor_alamano*(inside_sqrt[inside_sqrt > 0])**(1/4)

w1_alamano = z1_alamano / w1_factor_alamano

plt.figure()
ax = plt.gca()
ax.scatter(F_exp, Z_exp, label='z1')
ax.scatter(F_exp, W_exp, label='w1')
ax.plot(F_exp, F_exp, label='z0', color='gray', ls='--')
ax.plot(z0_test, z1_alamano, label='z1 alamano')
ax.plot(z0_test, w1_alamano, label='w1 alamano')
ax.set_xlabel('$z_0$ [mm]')
ax.set_ylabel('Amplitude [mm]')
ax.legend()


# <codecell>


eta = -0

muf = mu
muw = mu

F = (2 * np.pi * f0) ** 3 * (z0test / 2) ** 2 / (muw * muf * w0 * vc)

z0seuil = z0test[np.argmin((F + eta)**2)]

Delta = - (muf/2)*(muw/2) * (eta + F) # matrix determinant
tr = -muw/2 + eta * muf/2 # matrix trace

D = tr**2 - 4*Delta # system determinant
lam = (tr + np.sqrt(D))/2 # biggest eigenvalue

den = muw + 2*lam
den[den==0] = 1e-10
fact = (2 * np.pi * f0) / den

ratio = np.sqrt(w0 * vc / (2 * np.pi * f0) * (-eta * mu + 2 * lam) / (mu + 2 * lam))




# <codecell>

crit = instab & (F_exp > 0.253)


ratio_th = np.sqrt(w0 * vc / omega0)
ratio_th_u = ratio_th * np.sqrt((1/2 * w0_u / w0) ** 2 + (1/2 * vc_u / vc) ** 2)


### WITH THE LAMBDA
eta = 0

F = (omega0) ** 3 * (z0test / 2) ** 2 / (mu * mu * w0 * vc) # order parameter

def lam(eta_):
    Delta = - (mu/2)*(mu/2) * (eta_ + F) # matrix determinant
    tr = -mu/2 + eta_ * mu/2 # matrix trace
    D = tr**2 - 4*Delta # system determinant
    return (tr + np.sqrt(D))/2 # biggest eigenvalue

ratio_fin_min = ratio_th * np.sqrt((-(0)*mu + 2*lam(0)) / (mu + 2*lam(0))) # eta = 0
ratio_fin_max = ratio_th * np.sqrt((-(-1)*mu + 2*lam(-1)) / (mu + 2*lam(-1))) # eta = -1
ratio_fin = (ratio_fin_max + ratio_fin_min)/2
ratio_fin_u = (ratio_fin_max - ratio_fin_min)/2



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




# <codecell>



