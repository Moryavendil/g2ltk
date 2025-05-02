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

data_sheetname = 'autodata2'

df_autodata = pd.read_excel(workbookpath, sheet_name=data_sheetname)
datakeys = [key for key in df_autodata.keys() if 'unnamed' not in key.lower()]

datavalid = df_autodata['F_f'].astype(str) != 'nan'
instab = df_autodata['Z_a'].astype(str) != 'nan'

N_acq = np.sum(datavalid)

acquisition_title = df_autodata['acquisition_title'][datavalid].to_numpy()
F_f = df_autodata['F_f'][datavalid].to_numpy() * fr_per_s
Z_f = df_autodata['Z_f'][datavalid].to_numpy() * fr_per_s
W_f = df_autodata['W_f'][datavalid].to_numpy() * fr_per_s
F_q = df_autodata['F_q'][datavalid].to_numpy() * px_per_mm
Z_q = df_autodata['Z_q'][datavalid].to_numpy() * px_per_mm
W_q = df_autodata['W_q'][datavalid].to_numpy() * px_per_mm
F_a = df_autodata['F_a'][datavalid].to_numpy() / px_per_mm
Z_a = df_autodata['Z_a'][datavalid].to_numpy() / px_per_mm
W_a = df_autodata['W_a'][datavalid].to_numpy() / px_per_mm
F_p = df_autodata['F_p'][datavalid].to_numpy()
Z_p = df_autodata['Z_p'][datavalid].to_numpy() 
W_p = df_autodata['W_p'][datavalid].to_numpy()

color_q = utility.color_q
color_w = utility.color_w
color_z = utility.color_z
color_f = 'k'


# <codecell>

fig, axes = plt.subplots(1,1, figsize=utility.figsize('simple'), squeeze=False, sharex=True)
ax = axes[0,0]
ax.scatter(m_excitation_amplitude, F_a, label='exp pts', color=color_f)
# ax.plot(Vrpz, z0_th(Vrpz, *popt))
ax.set_xlabel('Driving signal amplitude [mV]')
ax.set_ylabel('$z_0$ [mm]')


# <codecell>

fig, axes = plt.subplots(2, 1, figsize=utility.figsize('wide', ratio = .8), sharex=True, squeeze=False)
errorbar_kw = {'capsize': 3, 'ls': '', 'marker':'o', 'mfc': 'w'}

ax = axes[0,0]
ax.scatter(m_excitation_amplitude, F_f, color=color_f)
ax.scatter(m_excitation_amplitude[instab], Z_f[instab], color=color_z)
ax.scatter(m_excitation_amplitude[instab], W_f[instab], color=color_w)

ax.set_ylabel('Frequency [Hz]')
ax.set_xlabel('Driving signal amplitude [mV]')

ax = axes[1,0]
ax.scatter(m_excitation_amplitude, F_q, color=color_f)
ax.scatter(m_excitation_amplitude[instab], Z_q[instab], color=color_z)
ax.scatter(m_excitation_amplitude[instab], W_q[instab], color=color_w)

ax.set_ylabel('Wavenumber [mm$^{-1}$]')
ax.set_xlabel('Driving signal amplitude [mV]')


# <codecell>

q0_exp = (W_q + Z_q)/2
f0_exp = 40

k0_exp = 2 * np.pi * q0_exp
omega0_exp = 2 * np.pi * f0_exp


# incert_q = np.zeros_like(F_q)
# incert_Z = np.zeros_like(Z_a)
# incert_W = np.zeros_like(W_a)

incert_q = 1 / 10 * q0_exp / 4
incert_Z = np.full_like(Z_a, mm_per_px / 4)
incert_W = np.full_like(W_a, mm_per_px / 4)
incert_Z[instab] += Z_a[instab] * .01
incert_W[instab] += W_a[instab] * .01

ratio_exp = W_a / Z_a / k0_exp
ratio_exp_u = ratio_exp * np.sqrt((incert_Z / Z_a) ** 2 + (incert_W / W_a) ** 2 + (incert_q / q0_exp) ** 2)


# <codecell>

### ATTENUATION
b = .58
b_u = 0.02
nu = 1.
mu = 1 / (b ** 2 / (12 * nu))
mu_u = mu * 2*np.sqrt((b_u/b)**2)
print(f'mu = {round(mu, 3)} pm {round(mu_u, 3)} Hz')

### BULK SPEED
g = 9.81e3
u0 = g / mu
u0_u = u0 * mu_u/mu
print(f'u0 = {round(u0, 3)} pm {round(u0_u, 3)} mm/s')

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
fz_v0 = 0.2
omegaz_v0 = 2*np.pi*fz_v0
omegaw_v0 = omegaz_v0 + omega0
k_v0 = omegaw_v0 / u0 

fz_v0 = omegaz_v0 / (2*np.pi)
fw_v0 = omegaw_v0 / (2*np.pi)
q_v0 = k_v0 / (2*np.pi)
q_v0_u = q_v0 * u0_u/u0

Fc_0 = 0.255

ETA = 1. 
ratio_v0 = np.sqrt(ETA) * np.sqrt(u0 ** 2 * w0 / (omega0 * vc))
ratio_v0_u = ratio_v0 * np.sqrt((w0_u/w0)**2 + (2*u0_u/u0)**2)


print(f'ratio [mm] = {round(ratio_v0, 3)} pm {round(ratio_v0_u, 3)} mm')
# print(f'ratio [mm] = {round(k_v0 * Fc_0/Z_0/phi_v0, 3)} pm {round(ratio_v0_u, 3)} mm')


# <codecell>

deltaphase_exp = (W_p[instab] - Z_p[instab]-F_p[instab])
deltaphase_exp[deltaphase_exp >= np.pi] -= 2*np.pi 


# <codecell>

logging.set_verbose('debug')
utility.activate_saveplot(False, style='jfm')


# <codecell>

fig, axes = plt.subplots(2, 1, figsize=utility.figsize('double'), sharex=True, squeeze=False)
errorbar_kw = {'capsize': 3, 'ls': '', 'mfc': 'w'}

ax = axes[0,0]
# ax.set_title('Mode structure: phase')
ax.errorbar(F_a[instab], deltaphase_exp, marker='o', label=r'$\arg(W) - \arg(F) - \arg(Z)$', color=color_q, **errorbar_kw)
ax.axhline(-np.pi/2, ls='-', label='Prediction: $\pi/2$', alpha=.6, color=color_q)

ax.axvline(Fc_0, color='k', ls=':')
ax.text(Fc_0, 0., 'Threshold')

utility.set_yaxis_rad(ax)
ax.set_ylabel(r'Phase [rad]')
ax.legend()

ax = axes[1,0]
# ax.set_title('Mode structure: amplitude')
exppts = ax.errorbar(F_a[instab], ratio_exp[instab], yerr=(np.minimum(ratio_exp, ratio_exp_u)[instab], ratio_exp_u[instab]), ms=4.5, marker='D', color=color_q, label=r'$|W| / (k\,|Z|)$', **errorbar_kw)
th_line = ax.axhline(ratio_v0, ls='-', label='Prediction: $u_0\,\sqrt{w_0 / (\omega_0\,v_c)}$', alpha=.6, color=color_q)
th_error = ax.axhspan(ratio_v0-ratio_v0_u, ratio_v0+ratio_v0_u, alpha=.1, zorder=.5, color=color_q, label=th_line.get_label())

ax.axvline(Fc_0, color='k', ls=':')
ax.text(Fc_0, 0.5, 'Threshold')

ax.set_ylim(0, 1.25)
ax.set_ylabel(r'... [mm]')
# handles, labels = ax.get_legend_handles_labels()
# handles = [handles[-1], (handles[0], handles[1])]
# labels = [labels[-1, labels[0]]]
ax.legend([exppts, (th_line, th_error)], [exppts.get_label(), th_line.get_label()])

axes[-1, 0].set_xlim(.15, .42)
axes[-1, 0].set_xlabel('Transverse movement amplitude $|F|$ [mm]')

# utility.save_graphe('modestructure_v1')


# <codecell>

fig, axes = plt.subplots(2, 1, figsize=utility.figsize('double'), sharex=True, squeeze=False)
errorbar_kw = {'capsize': 3, 'ls': '', 'marker':'o', 'mfc': 'w'}


ax = axes[0, 0]
ax.set_title('q(F)')
ax.errorbar(F_a[instab], q0_exp[instab], yerr=incert_q[instab], color=color_q, label=r'$q$ (exp.)', **errorbar_kw)
ax.axhline(q_v0, ls='--', label='$\omega_w/ v_w = f_0 / u_0$', alpha=.6, color=color_q)
ax.axhspan(q_v0-q_v0_u, q_v0+q_v0_u, alpha=.1, zorder=.5, color=color_q)

ax.set_ylim(0, .25)
ax.set_ylabel('$k/(2\pi)$ [mm$^{-1}$]')
ax.legend()


ax = axes[1,0]
ax.scatter(F_a, Z_a, label=r'$|{Z}|$ (exp.)', color=color_z)
ax.scatter(F_a, W_a, label=r'$|{W}|$ (exp.)', color=color_w)
ax.plot(F_a, F_a, label=r'$|{F}|$', color='k', alpha=.3, ls=':')
ax.axhspan(w0-w0_u, w0+w0_u, color='k', alpha=.1, zorder=.5)
ax.axhline(w0, color='k', alpha=.3, ls=':')

ax.set_ylabel('Amplitude [mm]')
ax.legend()


# <codecell>

Ftest = np.linspace(0.1, 0.45, 500)

# NOW THE FIT


def Z_fit(F, q0, qZ, Fc, ZF):
    return ZF*np.maximum(F - Fc, 0)**(1/4)

def q_fit(F, q0, qZ, Fc, ZF):
    return q0 + qZ*Z_fit(F, q0, qZ, Fc, ZF)**2

def W_fit(F, q0, qZ, Fc, ZF):
    return ratio_v0 * Z_fit(F, q0, qZ, Fc, ZF) * 2*np.pi*q_fit(F, q0, qZ, Fc, ZF)

def minfn(params):
    # The references (uncertainties)
    # q_ref = q_v0
    # Z_ref = Z_0
    q_ref = 0.01
    Z_ref = 0.05
    W_ref = ratio_v0 * Z_ref * 2*np.pi*q_v0
    
    DeltaZ = Z_a[valid_for_fit] - Z_fit(F_a[valid_for_fit], *params)
    DeltaW = W_a[valid_for_fit] - W_fit(F_a[valid_for_fit], *params)
    Deltaq = q0_exp[valid_for_fit] - q_fit(F_a[valid_for_fit], *params)
    return np.sum( (DeltaZ / Z_ref)**2 + (DeltaW / W_ref)**2 + (Deltaq / q_ref)**2 )

Fc_guess = Fc_0
q0_guess = q_v0
qZ_guess = .3
ZF_guess = .6

p_guess = [q0_guess, qZ_guess, Fc_guess, ZF_guess]


# <codecell>

valid_for_fit = F_a > 0.265
valid_for_fit = F_a > 0.18


from scipy.optimize import minimize

params_fitted = minimize(minfn, p_guess).x
# FcAB = [Fc_guess, *minimize(minfn_fixedthreshold, p_guess[1:]).x]


# <codecell>

from matplotlib.markers import MarkerStyle

fig, axes = plt.subplots(2, 1, figsize=utility.figsize('double'), sharex=True, squeeze=False)
errorbar_kw = {'capsize': 3, 'ls': '', 'marker':'o'}

ax = axes[0, 0]
ax.errorbar(F_a[instab], q0_exp[instab], yerr=incert_q[instab], color=color_q, label=r'$q$ (exp.)', mfc='w', **errorbar_kw)
ax.axhline(q_v0, ls='--', label=r'Linear prediction', alpha=.6, color=color_q)
ax.axhspan(q_v0-q_v0_u, q_v0+q_v0_u, alpha=.1, zorder=.5, color=color_q)
ax.plot(Ftest, q_fit(Ftest, *params_fitted), ls='-', label='fit', alpha=.8, color=color_q)

ax.axvline(Fc_0, color='k', ls=':')
ax.text(Fc_0, 0.15, 'Threshold')

ax.axvline(0.415, color='k', ls=':')
ax.text(0.415, 0.15, 'Break')

# ax.set_ylim(0, .2)
ax.set_ylim(.1, .2)
ax.set_ylabel('$k/(2\pi)$ [mm$^{-1}$]')
ax.legend()
# markerfacecolor='tab:blue',
markerfacecoloralt=color_z
# markeredgecolor='brown'

ax = axes[1,0]
# ax.errorbar(F_a, Z_a, color='w', markeredgewidth=.5, markerfacecolor='#00000000', markerfacecoloralt=color_z, fillstyle='right', label=r'$|{Z}|$ (exp.)', **errorbar_kw)
# ax.errorbar(F_a, W_a, color='w', markeredgewidth=.5, markerfacecolor='#00000000', markerfacecoloralt=color_w, fillstyle='left', label=r'$|{W}|$ (exp.)', **errorbar_kw)
# ax.errorbar(F_a, Z_a, color='k', markeredgewidth=1, fillstyle='none', **errorbar_kw)
# ax.errorbar(F_a, W_a, color='k', markeredgewidth=1, fillstyle='none', **errorbar_kw)
ax.errorbar(F_a, Z_a, color=color_z, mfc='w', label=r'$|{Z}|$ (exp.)', **errorbar_kw)
ax.errorbar(F_a, W_a, color=color_w, mfc='w', label=r'$|{W}|$ (exp.)', **errorbar_kw)
ax.plot(Ftest, Z_fit(Ftest, *params_fitted), ls='-', label='fit', alpha=.8, color=color_z)
ax.plot(Ftest, W_fit(Ftest, *params_fitted), ls='-', label='fit', alpha=.8, color=color_w)
# ax.plot(F_a, F_a, label=r'$|{F}|$', color='k', alpha=.3, ls=':')

ax.axhspan(w0-w0_u, w0+w0_u, color='k', alpha=.1, zorder=.5)
ax.axhline(w0, color='k', alpha=.3, ls=':')
ax.text(0.32, w0, r'$w_0$')


ax.axvline(Fc_0, color='k', ls=':')
ax.text(Fc_0, 0.2, 'Threshold')

ax.axvline(0.415, color='k', ls=':')
ax.text(0.415, 0.2, 'Break')

ax.set_ylim(0, .45)
ax.set_ylabel('Amplitude [mm]')
ax.legend()

ax.set_xlim(.15, .42)
ax.set_xlabel('Forcing amplitude $|F|$ [mm]')

utility.save_graphe('detuning-amplitude_v1')


# <codecell>

params_fitted


# <codecell>

utility.deactivate_saveplot(style='confort')


# <codecell>

fig, axes = plt.subplots(2, 1, figsize=utility.figsize('wide', ratio=1), sharex=True, squeeze=False)
errorbar_kw = {'capsize': 3, 'ls': '', 'marker':'o', 'mfc': 'w'}

critt = F_a > Fc_0

ax = axes[0, 0]
ax.set_title('q(F)')
ax.errorbar(F_a[critt]-Fc_0, (q0_exp[critt]-params_fitted[0]), color=color_q, label=r'$q$ (exp.)', **errorbar_kw)

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_aspect('equal')
ax.set_xlim(1e-3, 2e-1)
ax.set_ylim(1e-3, 5e-2)
ax.legend()


ax = axes[1,0]
ax.scatter(F_a[critt]-Fc_0, Z_a[critt], label=r'$|{Z}|$ (exp.)', color=color_z)
ax.scatter(F_a[critt]-Fc_0, W_a[critt], label=r'$|{W}|$ (exp.)', color=color_w)
fig, axes = plt.subplots(2, 1, figsize=utility.figsize('wide'), sharex=True, squeeze=False)
errorbar_kw = {'capsize': 3, 'ls': '', 'marker': 'o', 'mfc': 'w'}

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_aspect('equal')
ax.set_xlim(1e-3, 2e-1)
ax.set_ylim(1e-2, .5)
ax.legend()


# <codecell>




# <codecell>




# <codecell>



