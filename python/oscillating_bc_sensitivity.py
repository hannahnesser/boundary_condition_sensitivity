import math
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
import gcpy as gc
import inversion as inv
import settings as s
import plot
import plot_settings as ps
ps.SCALE = ps.PRES_SCALE
ps.BASE_WIDTH = ps.PRES_WIDTH
ps.BASE_HEIGHT = ps.PRES_HEIGHT
import format_plots as fp

rcParams['text.usetex'] = True
np.set_printoptions(precision=3, linewidth=300, suppress=True)

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
# Default is opt_BC = False
U = np.concatenate([np.arange(5, 0, -1), 
                    np.array([0.1, -0.1]), 
                    np.arange(-1, -5, -1)])*24
U = np.repeat(U, 2)
U = 5*24

true = inv.Inversion(gamma=1, U=U)
# true_opt = inv.Inversion(opt_BC=True, gamma=1, U=U)

## -------------------------------------------------------------------------##
# Define a standard oscillating perturbation
## -------------------------------------------------------------------------##
def oscillating_bc_pert(t, y, amp, freq, phase):
    return y + amp*np.sin((2*np.pi/true.t.max())*freq*(t + phase))

# y-intercept, amplitude, frequency, phase
BC_pert_std = [true.BC, 10, 2, 0]
BC_pert_std_v = oscillating_bc_pert(true.t, *BC_pert_std)
pert_std = inv.Inversion(BC=BC_pert_std_v, gamma=1, U=U)
pert_std_opt = inv.Inversion(BC=BC_pert_std_v, opt_BC=True, gamma=1, U=U)

## -------------------------------------------------------------------------##
# Test the effect of changed y-intercept, amplitude, frequency, and phase 
## -------------------------------------------------------------------------##
figsize = fp.get_figsize(aspect=2.5, rows=4, cols=6, 
                         max_width=ps.BASE_WIDTH,
                         max_height=ps.BASE_HEIGHT)
fig = plt.figure(figsize=figsize)
gs = gridspec.GridSpec(4, 8, width_ratios=[1, 0.2, 1, 1, 1, 1, 1, 1])
ax = []
for j in [0, 2, 3, 4, 5, 6, 7]:
    ax_i = [fig.add_subplot(gs[0, j])]
    for i in range(1, 4):
        if j <= 2:
            sharey = ax_i[0]
        elif j > 2:
            sharey = ax[1][0]
        ax_i.append(fig.add_subplot(gs[i, j], sharey=sharey,
                                          sharex=ax_i[0]))
    ax.append(ax_i)
ax = np.array(ax).T

# fig, ax = fp.get_figax(rows=4, cols=4, aspect=2.5, 
#                              sharex='col', sharey='col', 
#                              max_width=c.BASE_WIDTH*3, 
#                              max_height=c.BASE_HEIGHT*3)

fp.add_title(ax[0, 0], 'Boundary condition', fontsize=ps.LABEL_FONTSIZE*1.5)
fp.add_title(ax[0, 1], 'Standard inversion', 
             fontsize=ps.LABEL_FONTSIZE*1.5)
fp.add_title(ax[0, 2], 
            'Inflated prior errors\nin first grid cell', 
             fontsize=ps.LABEL_FONTSIZE*1.5)
fp.add_title(ax[0, 3], 
             'Inflated observing system\nerrors in first grid cell',
             fontsize=ps.LABEL_FONTSIZE*1.5)
fp.add_title(ax[0, 4], 
            'Boundary condition\noptimized', 
             fontsize=ps.LABEL_FONTSIZE*1.5)
fp.add_title(ax[0, 5], 
            'Inflated prior errors\nin first grid cell and\nboundary condition\noptimized',
             fontsize=ps.LABEL_FONTSIZE*1.5)
fp.add_title(ax[0, 6], 
            'Boundary condition\noptimized in chunks',
             fontsize=ps.LABEL_FONTSIZE*1.5)

ys = np.arange(1900, 1950, 10)
amps = np.arange(10, 60, 10)
freqs = np.arange(1, 6)
phase = np.pi*np.arange(0, 2, 0.4)
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
    ax[j, 0].axhline(true.BC_t, color='0.6', ls='--', zorder=-1, lw=0.75)

    # Perturbation
    ax[j, 0].fill_between(true.obs_t, 1800, 2000, color='0.6', alpha=0.3)
    ax[j, 0].set_xlim(true.t.min(), true.t.max())
    ax[j, 0].set_ylim(1850, 1975)
    ax[j, 0].set_ylabel('Boundary\ncondition (ppb)')

    # Labels
    ax[j, 1].set_ylabel(r'$\Delta\hat{x}$')

    for i, val in enumerate(vals):
        BC_pert = dc(BC_pert_std)
        BC_pert[j] = val
        BC_pert = oscillating_bc_pert(true.t, *BC_pert)
        ax[j, 0].plot(true.t, BC_pert, 
            color=fp.color(2*i, lut=12), lw=0.75)

        # Non optimized BC
        pert = inv.Inversion(BC=BC_pert, gamma=1, U=U)
        ils_j.append(pert.ils)
        ax[j, 1].plot(pert.xp, np.abs((pert.xhat - true.xhat)/true.xhat), 
                      color=fp.color(2*i, lut=12), ls='-', lw=0.75,
                      marker='o', ms=2)
        ax_1 = ax[j, 1].twinx()
        ax_1.set_ylim(1, 0)
        # test = pert.xhat*pert.xhat/(pert.xhat - pert.bc_contrib)
        # ax_1.plot(pert.xp, (test - true.xhat)/true.xhat,
        #           color=fp.color(2*i, lut=12), ls=':')
        ax_1.plot(pert.xp, (pert.bc_contrib/pert.xhat),
                  color=fp.color(2*i, lut=12), ls='--', lw=0.75)
        # ax_1.plot(pert.xp, np.diagonal(pert.a),
        #          color=fp.color(2*i, lut=12), ls=':', lw=1)
        # ax[j, 1].axvline(pert.ils + 0.5, color='darkgreen', ls=':', lw=0.75)
        plt.setp(ax_1.get_yticklabels(), visible=False)

        # Sa first element scaled
        sa_sf = dc(true.sa)
        sa_sf[0] *= 100
        pert_sa_BC = inv.Inversion(sa=sa_sf, BC=BC_pert, gamma=1, U=U)
        ax[j, 2].plot(pert_sa_BC.xp,
                      np.abs((pert_sa_BC.xhat - true.xhat)/true.xhat), 
                      color=fp.color(2*i, lut=12), ls='-', lw=0.75,
                      marker='o', ms=2)
        ax_2 = ax[j, 2].twinx()
        ax_2.set_ylim(1, 0)
        # test = (pert_sa_BC.xhat*pert_sa_BC.xhat
        #         /(pert_sa_BC.xhat - pert_sa_BC.bc_contrib))
        # ax_2.plot(pert_sa_BC.xp, (test - true.xhat)/true.xhat,
        #           color=fp.color(2*i, lut=12), ls=':')
        ax_2.plot(pert_sa_BC.xp, (pert_sa_BC.bc_contrib/pert_sa_BC.xhat),
                     color=fp.color(2*i, lut=12), ls='--', lw=0.75)
        # ax_2.plot(pert_sa_BC.xp, np.diagonal(pert_sa_BC.a),
        #          color=fp.color(2*i, lut=12), ls=':', lw=1)
        # ax[j, 2].axvline(pert_sa_BC.ils + 0.5, color='darkgreen', ls=':', 
        #                 lw=0.75)
        plt.setp(ax_2.get_yticklabels(), visible=False)

        # So first element scaled
        so_sf = dc(true.so)
        so_sf[:s.nobs_per_cell] *= 100
        pert_so = inv.Inversion(so=so_sf, BC=BC_pert, opt_BC=False,
                                 gamma=1, U=U)
        ax[j, 3].plot(pert_so.xp, 
                     np.abs((pert_so.xhat - true.xhat)/true.xhat),
                     color=fp.color(2*i, lut=12), ls='-', lw=0.75,
                     marker='o', ms=2)
        ax_3 = ax[j, 3].twinx()
        ax_3.set_ylim(1, 0)
        ax_3.plot(pert_so.xp, (pert_so.bc_contrib/pert_so.xhat),
                  color=fp.color(2*i, lut=12), ls='--', lw=0.75)
        plt.setp(ax_3.get_yticklabels(), visible=False)

        # Optimized BC
        pert_opt = inv.Inversion(BC=BC_pert, opt_BC=True, gamma=1, U=U)
        ax[j, 4].plot(pert_opt.xp, 
                      np.abs((pert_opt.xhat - true.xhat)/true.xhat), 
                      color=fp.color(2*i, lut=12), ls='-', lw=0.75,
                      marker='o', ms=2)
        ax_4 = ax[j, 4].twinx()
        ax_4.set_ylim(1, 0)
        # test = (pert_opt.xhat*pert_opt.xhat
        #         /(pert_opt.xhat - pert_opt.bc_contrib))
        # ax_3.plot(pert_opt.xp, (test - true.xhat)/true.xhat,
        #           color=fp.color(2*i, lut=12), ls=':')
        ax_4.plot(pert_opt.xp, (pert_opt.bc_contrib/pert_opt.xhat),
                     color=fp.color(2*i, lut=12), ls='--', lw=0.75)
        # ax_3.plot(pert_opt.xp, np.diagonal(pert_opt.a),
        #          color=fp.color(2*i, lut=12), ls=':', lw=1)
        # ax[j, 3].axvline(pert_opt.ils + 0.5, color='darkgreen', ls=':', 
        #                     lw=0.75)
        plt.setp(ax_4.get_yticklabels(), visible=False)

        # SA first element scaled and BC optimized
        sa_sf = dc(true.sa)
        sa_sf[0] *= 10
        pert_sa_BC = inv.Inversion(sa=sa_sf, BC=BC_pert, opt_BC=true, gamma=1, U=U)
        ax[j, 5].plot(pert_sa_BC.xp,
                      np.abs((pert_sa_BC.xhat - true.xhat)/true.xhat), 
                      color=fp.color(2*i, lut=12), ls='-', lw=0.75,
                      marker='o', ms=2)
        ax_5 = ax[j, 5].twinx()
        ax_5.set_ylim(1, 0)
        # test = (pert_sa_BC.xhat*pert_sa_BC.xhat
        #         /(pert_sa_BC.xhat - pert_sa_BC.bc_contrib))
        # ax_5.plot(pert_sa_BC.xp, (test - true.xhat)/true.xhat,
        #           color=fp.color(2*i, lut=12), ls=':')
        ax_5.plot(pert_sa_BC.xp, 
                  (pert_sa_BC.bc_contrib/pert_sa_BC.xhat),
                  color=fp.color(2*i, lut=12), ls='--', lw=0.75)
        # ax_5.plot(pert_sa_BC.xp, np.diagonal(pert_sa_BC.a),
        #          color=fp.color(2*i, lut=12), ls=':', lw=1)
        # ax[j, 5].axvline(pert_sa_BC.ils + 0.5, color='darkgreen', ls=':', 
        #                     lw=0.75)
        # ax_5.set_ylabel(r'$\frac{\vert Gc \vert}{\vert Gc \vert + \vert A x_A \vert}$', rotation=270, labelpad=10, va='center')
        plt.setp(ax_5.get_yticklabels(), visible=False)

        # Multiple BC optimizaitons 
        nopt = 4
        pert_opt = inv.Inversion(
            BC=BC_pert, opt_BC=True, opt_BC_n=nopt, gamma=1, U=U)
        ax[j, 6].plot(pert_opt.xp, 
                      np.abs((pert_opt.xhat - true.xhat)/true.xhat), 
                      color=fp.color(2*i, lut=12), ls='-', lw=0.75,
                      marker='o', ms=2)
        ax_6 = ax[j, 6].twinx()
        ax_6.set_ylim(1, 0)
        # test = (pert_opt.xhat*pert_opt.xhat
        #         /(pert_opt.xhat - pert_opt.bc_contrib))
        # ax_6.plot(pert_opt.xp, (test - true.xhat)/true.xhat,
        #           color=fp.color(2*i, lut=12), ls=':')
        ax_6.plot(pert_opt.xp, (pert_opt.bc_contrib/pert_opt.xhat),
                     color=fp.color(2*i, lut=12), ls='--', lw=0.75)
        # ax_6.plot(pert_opt.xp, np.diagonal(pert_opt.a),
        #          color=fp.color(2*i, lut=12), ls=':', lw=1)
        ax_6.set_ylabel(r'$\vert Gc/\hat{x} \vert$', rotation=270, labelpad=10, va='center')

        # And plot the BC optimizations...
        # ax[j, 0].hlines(BC_pert[pert_opt.t > pert_opt.obs_t.min()].mean(),
        #                 xmin=pert_opt.obs_t.min(), xmax=pert_opt.obs_t.max(),
        #                 colors='white', lw=1)
        BC_chunk = math.ceil(len(BC_pert)/nopt)
        # BC_pert = BC_pert[true.t >= true.obs_t.min()]
        BC_pert_avg = np.nanmean(
            np.pad(
                BC_pert, (0, (BC_chunk - BC_pert.size % BC_chunk) % BC_chunk),
                mode='constant', constant_values=np.NaN).reshape(-1, BC_chunk),
            axis=1)
        k = int(0)# np.where(true.t >= true.obs_t.min())[0][0]
        m = int(0)
        while k < len(true.t):
            ax[j, 0].hlines(pert_opt.xhat_BC[m], 
                             xmin=pert_opt.t[k], 
                             xmax=pert_opt.t[np.minimum(k + BC_chunk, len(pert_opt.t) - 1)],
                             colors=fp.color(2*i, lut=12), ls='--', lw=0.75)
            ax[j, 0].hlines(BC_pert_avg[m], 
                             xmin=pert_opt.t[k], 
                             xmax=pert_opt.t[np.minimum(k + BC_chunk, len(pert_opt.t) - 1)],
                             colors=fp.color(2*i, lut=12), ls=':', lw=0.75)
            k += BC_chunk
            m += int(1)
        plt.setp(ax_5.get_yticklabels(), visible=False)


for i in range(1, 7):
    # ax[0, i].set_xlim(0, true.nstate)
    ax[0, i].set_ylim(0, 0.4)
    ax[3, i].set_xlabel('State vector element')
ax[-1, 0].set_xlabel('Time (hr)')

for i in range(4):
    for j in range(1, 7):
        ax[i, j].set_xticks(np.arange(0, s.nstate+1, 5))
        ax[i, j].set_xlim(0.5, s.nstate + 0.5)
        for k in range(3):
            ax[i, j].axvline((k + 1)*5 + 0.5, c=fp.color(1), alpha=0.2,
                                ls=':', lw=0.5)
        if j != 1:
            # Turn off y labels
            plt.setp(ax[i, j].get_yticklabels(), visible=False)
    if i < 3:
        for j in range(7):
            # Turn off x labels
            plt.setp(ax[i, j].get_xticklabels(), visible=False)

fp.save_fig(fig, plot_dir, 'oscillating_BC_SV')

