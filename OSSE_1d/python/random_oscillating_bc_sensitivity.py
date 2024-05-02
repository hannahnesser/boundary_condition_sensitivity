import numpy as np
import pandas as pd
from copy import deepcopy as dc
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import rcParams
from matplotlib.lines import Line2D

# Custom packages
import sys
sys.path.append('.')
import settings as s
import gcpy as gc
# import forward_model as fm
# import inversion as inv_f
import inversion as inv
import plot
import config
config.SCALE = config.PRES_SCALE
config.BASE_WIDTH = config.PRES_WIDTH
config.BASE_HEIGHT = config.PRES_HEIGHT
import format_plots as fp

rcParams['text.usetex'] = True
np.set_printoptions(precision=1, linewidth=300, suppress=True)

## -------------------------------------------------------------------------##
# File Locations
## -------------------------------------------------------------------------##
plot_dir = '../plots'
plot_dir = f'{plot_dir}/n{s.nstate}_m{s.nobs}'
if not os.path.exists(plot_dir):
    os.mkdir(plot_dir)

## -------------------------------------------------------------------------##
# Define the true inversion
## -------------------------------------------------------------------------##
true = inv.Inversion()
true_opt = inv.Inversion(opt_BC=True)

## -------------------------------------------------------------------------##
# Define a standard oscillating perturbation
## -------------------------------------------------------------------------##
def oscillating_bc_pert(t, y, amp, freq, phase):
    return y + amp*np.sin((2*np.pi/true.t.max())*freq*(t + phase))

# y-intercept, amplitude, frequency, phase
BC_pert_std = [1920, 10, 2, 0]
BC_pert_std_v = oscillating_bc_pert(true.t, *BC_pert_std)

# Random noise for all BC perturbations
noise = true.rs.normal(0, 5, BC_pert_std_v.shape)

# Standard
BC_pert_std_v += noise

# Base inversion objects
pert_std = inv.Inversion(BC=BC_pert_std_v)
pert_std_opt = inv.Inversion(BC=BC_pert_std_v, opt_BC=True)

## -------------------------------------------------------------------------##
# Test the effect of changed y-intercept, amplitude, frequency, and phase 
## -------------------------------------------------------------------------##
figsize = fp.get_figsize(aspect=2.5, rows=4, cols=4, 
                         max_width=config.BASE_WIDTH*3,
                         max_height=config.BASE_HEIGHT*3)
fig_sv = plt.figure(figsize=figsize)
gs = gridspec.GridSpec(4, 6, width_ratios=[1, 0.2, 1, 1, 1, 1])
ax_sv = []
for j in [0, 2, 3, 4, 5]:
    ax_sv_i = [fig_sv.add_subplot(gs[0, j])]
    for i in range(1, 4):
        if j <= 2:
            sharey = ax_sv_i[0]
        elif j > 2:
            sharey = ax_sv[1][0]
        ax_sv_i.append(fig_sv.add_subplot(gs[i, j], sharey=sharey,
                                          sharex=ax_sv_i[0]))
    ax_sv.append(ax_sv_i)
ax_sv = np.array(ax_sv).T

fp.add_title(ax_sv[0, 0], 'Boundary condition', 
             fontsize=config.LABEL_FONTSIZE*1.5)
fp.add_title(ax_sv[0, 1], 'Boundary condition\nnot optimized', 
             fontsize=config.LABEL_FONTSIZE*1.5)
fp.add_title(ax_sv[0, 2], 'Boundary condition\nnot optimized and\n'r'$x_{A,0}$ scaled by 10', 
             fontsize=config.LABEL_FONTSIZE*1.5)
fp.add_title(ax_sv[0, 3], 'Boundary condition\noptimized', 
             fontsize=config.LABEL_FONTSIZE*1.5)
fp.add_title(ax_sv[0, 4], 'Boundary condition\noptimized and\n'r'$x_{A,0}$ scaled by 10', 
             fontsize=config.LABEL_FONTSIZE*1.5)

ys = np.arange(1900, 1950, 10)
amps = np.arange(10, 60, 10)
freqs = np.arange(1, 6)
phase = np.pi*np.arange(0, 1, 0.2)
periodics = {'y intercept' : ys, 'Amplitude' : amps, 
             'Frequency' : freqs, 'Phase' : phase}
units = ['ppb', 'ppb', r'hr$^{-1}$', 'rad']

ils = pd.DataFrame(columns=['ys', 'amps', 'freqs', 'phase'])
ils_opt = pd.DataFrame(columns=['ys', 'amps', 'freqs', 'phase'])
for j, (label, vals) in enumerate(periodics.items()):
    ils_j = []
    ils_opt_j = []
    unit = units[j]

    # Standard perturbation
    ax_sv[j, 0].axhline(true.BC_t, color='0.6', ls='--', zorder=-1, lw=0.75)

    # Perturbation
    ax_sv[j, 0].fill_between(true.obs_t, 1800, 2000, color='0.6', alpha=0.3)
    ax_sv[j, 0].set_xlim(true.t.min(), true.t.max())
    ax_sv[j, 0].set_ylim(1850, 1975)
    ax_sv[j, 0].set_ylabel('Boundary\ncondition (ppb)')

    # Labels
    ax_sv[j, 1].set_ylabel(r'$\frac{\vert\hat{x} - \hat{x}_{true}\vert}{\hat{x}_{true}}$')

    for i, val in enumerate(vals):
        BC_pert = dc(BC_pert_std)
        BC_pert[j] = val
        BC_pert = oscillating_bc_pert(true.t, *BC_pert)
        BC_pert += noise
        ax_sv[j, 0].plot(true.t, BC_pert, color=fp.color(2*i, lut=10), lw=0.75)

        # Non optimized BC
        pert = inv.Inversion(BC=BC_pert)
        ils_j.append(pert.ils)
        ax_sv[j, 1].plot(s.xp, np.abs(pert.xhat - true.xhat)/true.xhat, 
                         color=fp.color(2*i, lut=10), ls='-', lw=0.75,
                         marker='o', ms=2)
        ax_sv_1 = ax_sv[j, 1].twinx()
        ax_sv_1.set_ylim(0, 1.05)
        ax_sv_1.plot(s.xp, pert.xa_contrib/pert.tot_correct,
                     color=fp.color(2*i, lut=10), ls='--', lw=0.75)
        ax_sv_1.axvline(pert.ils + 0.5, color='darkgreen', ls=':', lw=0.75)
        plt.setp(ax_sv_1.get_yticklabels(), visible=False)

        # xA first element scaled
        x_abs_t_sf = dc(true.x_abs_t)
        x_abs_t_sf[0] *= 10
        xa_abs_sf = dc(true.xa_abs)
        xa_abs_sf[0] *= 10
        pert_xa = inv.Inversion(x_abs_t=x_abs_t_sf, xa_abs=xa_abs_sf)
        pert_xa_BC = inv.Inversion(x_abs_t=x_abs_t_sf, xa_abs=xa_abs_sf, 
                                   BC=BC_pert)
        ax_sv[j, 2].plot(s.xp,
                         np.abs(pert_xa_BC.xhat - pert_xa.xhat)/pert_xa.xhat, 
                         color=fp.color(2*i, lut=10), ls='-', lw=0.75,
                         marker='o', ms=2)
        ax_sv_2 = ax_sv[j, 2].twinx()
        ax_sv_2.set_ylim(0, 1.05)
        ax_sv_2.plot(s.xp, pert_xa_BC.xa_contrib/pert_xa_BC.tot_correct,
                     color=fp.color(2*i, lut=10), ls='--', lw=0.75)
        ax_sv_2.axvline(pert_xa_BC.ils + 0.5, color='darkgreen', ls=':', 
                        lw=0.75)
        plt.setp(ax_sv_2.get_yticklabels(), visible=False)

        # Optimized BC
        pert_opt = inv.Inversion(BC=BC_pert, opt_BC=True)
        ils_opt_j.append(pert_opt.ils)
        ax_sv[j, 3].plot(s.xp, 
                         np.abs(pert_opt.xhat - true_opt.xhat)/true_opt.xhat, 
                         color=fp.color(2*i, lut=10), ls='-', lw=0.75,
                         marker='o', ms=2)
        ax_sv_3 = ax_sv[j, 3].twinx()
        ax_sv_3.set_ylim(0, 1.05)
        ax_sv_3.plot(s.xp, pert_opt.xa_contrib/pert_opt.tot_correct,
                     color=fp.color(2*i, lut=10), ls='--', lw=0.75)
        ax_sv_3.axvline(pert_opt.ils + 0.5, color='darkgreen', ls=':', lw=0.75)
        plt.setp(ax_sv_3.get_yticklabels(), visible=False)

        # xA first element scaled
        x_abs_t_sf = dc(true.x_abs_t)
        x_abs_t_sf[0] *= 10
        xa_abs_sf = dc(true.xa_abs)
        xa_abs_sf[0] *= 10
        pert_xa = inv.Inversion(x_abs_t=x_abs_t_sf, xa_abs=xa_abs_sf, 
                                opt_BC=True)
        pert_xa_BC = inv.Inversion(x_abs_t=x_abs_t_sf, xa_abs=xa_abs_sf, 
                                   BC=BC_pert, opt_BC=True)
        ax_sv[j, 4].plot(s.xp,
                         np.abs(pert_xa_BC.xhat - pert_xa.xhat)/pert_xa.xhat, 
                         color=fp.color(2*i, lut=10), ls='-', lw=0.75,
                         marker='o', ms=2)
        ax_sv_4 = ax_sv[j, 4].twinx()
        ax_sv_4.set_ylim(0, 1.05)
        ax_sv_4 .plot(s.xp, 
                         pert_xa_BC.xa_contrib/pert_xa_BC.tot_correct,
                         color=fp.color(2*i, lut=10), ls='--', lw=0.75)
        ax_sv_4.axvline(pert_xa_BC.ils + 0.5, color='darkgreen', ls=':', lw=0.75)
        ax_sv_4.set_ylabel(r'$\frac{\vert Gc \vert}{\vert Gc \vert + \vert A x_A \vert}$', rotation=270, labelpad=10, va='center')

for i in range(1, 5):
    ax_sv[0, i].set_ylim(0, 1.05)
    ax_sv[3, i].set_xlabel('State vector element')
ax_sv[-1, 0].set_xlabel('Time (hr)')

for i in range(4):
    for j in range(1, 5):
        ax_sv[i, j].set_xticks(np.arange(0, s.nstate+1, 5))
        ax_sv[i, j].set_xlim(0.5, s.nstate + 0.5)
        for k in range(3):
            ax_sv[i, j].axvline((k + 1)*5 + 0.5, c=fp.color(1), alpha=0.2,
                                ls=':', lw=0.5)
        if j != 1:
            # Turn off y labels
            plt.setp(ax_sv[i, j].get_yticklabels(), visible=False)
    if i < 3:
        for j in range(5):
            # Turn off x labels
            plt.setp(ax_sv[i, j].get_xticklabels(), visible=False)

fp.save_fig(fig_sv, plot_dir, 'random_oscillating_BC_SV')