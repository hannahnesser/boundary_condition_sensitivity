import numpy as np
import pandas as pd
from copy import deepcopy as dc
import os
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.lines import Line2D

# Custom packages
import sys
sys.path.append('.')
import settings as s
import gcpy as gc
import inversion as inv
import plot
import plot_settings as ps
ps.SCALE = ps.PRES_SCALE
ps.BASE_WIDTH = ps.PRES_WIDTH
ps.BASE_HEIGHT = ps.PRES_HEIGHT
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
# Define sensitivity test parameters
## -------------------------------------------------------------------------##
# Default is opt_BC = False
true_BC = inv.Inversion()
print(true_BC.t)
print(true_BC.obs_t)

## -------------------------------------------------------------------------##
# ILS sensitivity perturbations
## -------------------------------------------------------------------------##
fig_ils, ax_ils = fp.get_figax(rows=1, cols=3, aspect=1, 
                               sharex=False, sharey=True)
plt.subplots_adjust(wspace=0.1, hspace=0.5)

fig_summ, ax_summ = fp.get_figax(rows=3, cols=1, aspect=3,
                                 # width_ratios=[0.25, 1, 1],
                                 sharex=True, sharey='col')
plt.subplots_adjust(wspace=0.5, hspace=0.5)
ax_summ[0].set_ylim(0, 1.05)
ax_summ_t = [ax_summ[i].twinx() for i in range(3)]

fig_bc, ax_bc = fp.get_figax(rows=1, cols=3, aspect=1, sharey=True)
plt.subplots_adjust(wspace=0.1, hspace=0.1)

# ils_sensi = pd.DataFrame(columns=np.arange(1, 10))

# BC perturbations
perts = np.arange(5, 55, 10)
BC_ils = []
for i, pert in enumerate(perts):
    pert_BC = inv.Inversion(BC=true_BC.BC_t + pert)
    BC_ils.append(pert_BC.ils)
    ax_summ[0].plot(pert_BC.xp, 
                    np.abs(pert_BC.xhat - true_BC.xhat)/true_BC.xhat,
                    color=fp.color(2*i, lut=10), lw=2,
                    label=f'{pert} ppb')
    ax_summ_t[0].plot(pert_BC.xp, pert_BC.xa_contrib/pert_BC.tot_correct,
                  color=fp.color(2*i, lut=10), ls='--', lw=2)


    # t = true_BC.x_abs_t/true_BC.xa_abs
    # ax_summ[0, 1].plot(
    #     c.xp, 
    #     np.abs(pert_BC.xhat - t)/t,
    #     color=fp.color(2*i, lut=10), lw=2)
    # # ax_summ[0, 1].plot(
    # #     c.xp, pert_BC.xa_contrib/pert_BC.tot_correct,
    # #     color=fp.color(2*i, lut=10), ls='--', lw=2)

    ax_bc[0].plot(pert_BC.bc_contrib/pert_BC.tot_correct, 
                  np.abs(pert_BC.xhat - true_BC.xhat)/true_BC.xhat,
                  color=fp.color(2*i, lut=10), lw=0.5, ms=5, marker='o')

fp.add_title(ax_bc[0], 'Boundary\ncondition\nperturbations')
ax_bc[0].axvline(0.5, color='0.6', ls='--')
ax_bc[0].axhline(0.1, color='0.6', ls='--')

# ax_summ[0].set_xlabel('Rela')

# ax_summ[0].set_ylabel()
ax_ils[0].scatter(perts, BC_ils, color=fp.color(4), s=5)
ax_ils[0].set_xlim(perts.min() - 0.5, perts.max() + 0.5)

fp.add_title(ax_summ[0], 'Boundary condition perturbations')
fp.add_legend(ax_summ[0], bbox_to_anchor=(1.15, 0.5), loc='center left')

# Scale U
us = 24*np.arange(1, 11, 2)
U_ils = []
for i, u in enumerate(us):
    cc = fp.color(2*i, lut=10)
    true_U = inv.Inversion(U=u)
    pert_U = inv.Inversion(U=u, BC=true_BC.BC_t + 10)
    U_ils.append(true_U.ils)
    ax_summ[1].plot(pert_U.xp, np.abs(pert_U.xhat - true_U.xhat)/true_U.xhat, 
                    color=cc, lw=2, label=f'{(pert_U.L/u):.1f} day')
    ax_summ_t[1].plot(pert_U.xp, pert_U.xa_contrib/pert_U.tot_correct,
                      color=cc, ls='--', lw=2)

    # t = true_U.x_abs_t/true_U.xa_abs
    # ax_summ[1, 1].plot(
    #     c.xp, 
    #     np.abs(pert_U.xhat - t)/t,
    #     color=cc, lw=2)
    # # ax_summ[1, 1].plot(
    # #     c.xp, pert_U.xa_contrib/pert_U.tot_correct,
    # #     color=cc, ls='--', lw=2)


    ax_bc[1].plot(pert_U.bc_contrib/pert_U.tot_correct, 
                     np.abs(pert_U.xhat - true_U.xhat)/true_U.xhat,
                     color=cc, lw=0.5, ms=5, marker='o')


fp.add_title(ax_bc[1], 'Residence time\nchanges')
ax_bc[1].axvline(0.5, color='0.6', ls='--')
ax_bc[1].axhline(0.1, color='0.6', ls='--')

ax_ils[1].scatter(true_U.L/us, U_ils, color=fp.color(4), s=5)
ax_ils[1].set_xlim(true_U.L/us.max() - 0.5, true_U.L/us.min() + 0.5)

fp.add_title(ax_summ[1], 'Changing residence time')
fp.add_legend(ax_summ[1], bbox_to_anchor=(1.15, 0.5), loc='center left')

# # xa_abs[0] scaling
sfs = np.array([1, 2, 3, 4, 5])
# xa_abs_ils =[]
# for i, sf in enumerate(sfs):
#     # x_abs_t_sf = dc(true_BC.x_abs_t)
#     # x_abs_t_sf[0] *= sf
#     xa_abs_sf = dc(true_BC.xa_abs)
#     xa_abs_sf[0] *= sf
#     # pert_xa = inv.Inversion(x_abs_t=x_abs_t_sf, xa_abs=xa_abs_sf)
#     # pert_xa_BC = inv.Inversion(x_abs_t=x_abs_t_sf, xa_abs=xa_abs_sf,
#     #                            BC=true_BC.BC_t + 10)
#     pert_xa = inv.Inversion(xa_abs=xa_abs_sf)
#     pert_xa_BC = inv.Inversion(xa_abs=xa_abs_sf, BC=true_BC.BC_t + 10)
#     xa_abs_ils.append(pert_xa.ils)
#     ax_summ[2, 0].plot(c.xp, 
#                        np.abs(pert_xa_BC.xhat - pert_xa.xhat)/pert_xa.xhat, 
#                        color=fp.color(2*i, lut=10), lw=2)
#     ax_summ[2, 0].plot(c.xp, pert_xa_BC.xa_contrib/pert_xa_BC.tot_correct,
#                        color=fp.color(2*i, lut=10), ls='--', lw=2)

#     t = true_BC.x_abs_t/true_BC.xa_abs
#     ax_summ[2, 1].plot(
#         c.xp, 
#         np.abs(pert_xa_BC.xhat - t)/t,
#         color=fp.color(2*i, lut=10), lw=2)
#     ax_summ[2, 1].plot(
#         c.xp, pert_xa_BC.xa_contrib/pert_xa_BC.tot_correct,
#         color=fp.color(2*i, lut=10), ls='--', lw=2)


#     ax_bc[1, 0].plot(pert_xa_BC.bc_contrib/pert_xa_BC.tot_correct, 
#                      np.abs(pert_xa_BC.xhat - pert_xa.xhat)/pert_xa.xhat,
#                      color=fp.color(2*i, lut=10), lw=0.5, ms=5, marker='o')

# ax_bc[1, 0].text(0.05, 0.95, r'$x_{A,0}$ scaling',  va='top',
#                  fontsize=c.LABEL_FONTSIZE*c.SCALE,
#                  transform=ax_bc[1, 0].transAxes)
# ax_bc[1, 0].axvline(0.5, color='0.6', ls='--')
# ax_bc[1, 0].axhline(0.1, color='0.6', ls='--')


# ax_ils[1, 0].scatter(sfs, xa_abs_ils, color=fp.color(4), s=5)
# ax_ils[1, 0].set_xlim(sfs.min() - 0.5, sfs.max() + 0.5)

# Sa_abs scaling
sa_ils =[]
for i, sf in enumerate(sfs):
    sa_sf = dc(true_BC.sa)
    sa_sf[0] *= sf**2
    pert_sa = inv.Inversion(sa=sa_sf)
    pert_sa_BC = inv.Inversion(sa=sa_sf, BC=true_BC.BC_t + 10)
    sa_ils.append(pert_sa.ils)
    ax_summ[2].plot(pert_sa_BC.xp, 
                    np.abs(pert_sa_BC.xhat - true_BC.xhat)/true_BC.xhat, 
                    color=fp.color(2*i, lut=10), lw=2, label=f'{sf}')
    ax_summ_t[2].plot(pert_sa_BC.xp, 
                      pert_sa_BC.xa_contrib/pert_sa_BC.tot_correct,
                      color=fp.color(2*i, lut=10), ls='--', lw=2)

    ax_bc[2].plot(pert_sa_BC.bc_contrib/pert_sa_BC.tot_correct, 
                  np.abs(pert_sa_BC.xhat - true_BC.xhat)/true_BC.xhat,
                  color=fp.color(2*i, lut=10), lw=0.5, ms=5, marker='o')

ax_ils[2].scatter(sfs, sa_ils, color=fp.color(4), s=5)
ax_ils[2].set_xlim(sfs.min() - 0.5, sfs.max() + 0.5)

fp.add_title(ax_bc[2], r'$\sigma_{A,0}$ scaling')
ax_bc[2].axvline(0.5, color='0.6', ls='--')
ax_bc[2].axhline(0.1, color='0.6', ls='--')

fp.add_title(ax_summ[2], r'Scaling $\sigma_{A,0}$')
fp.add_legend(ax_summ[2], bbox_to_anchor=(1.15, 0.5), loc='center left')

# # Formatting ax_summ
# for i in range(3):
#     fp.add_labels(ax_summ[i, 0], '', '')
# # for ax in ax_summ.flatten():
# #     ax = fp.format_
for i in range(3):
    ax_summ[i].set_xticks(np.arange(0, s.nstate+1, 5))
    ax_summ[i].set_xlim(0.5, s.nstate + 0.5)
    ax_summ_t[i].set_ylim(0, 1.05)
    for k in range(21):
        ax_summ[i].axvline(k + 0.5, c=fp.color(1), alpha=0.2,
                            ls=':', lw=0.5)
    if i == 2:
        xlabel = 'State vector element'
    else:
        xlabel = ''

    fp.add_labels(ax_summ[i], xlabel, r'$\Delta x$')
        # r'$\frac{\vert\hat{x} - \hat{x}_{T}\vert}{\hat{x}_{T}}$')
    # fp.add_labels(
    #     ax_summ[i, 1],
    #     x    r'$\frac{\vert\hat{x} - x_{T}\vert}{x_{T}}$')

    fp.add_labels(ax_summ_t[i], '', r'$\zeta^{-1}$')

# Formatting ax_ils
ax_ils[2].set_yticks(np.arange(0, 10))
ax_ils[2].set_ylim(0.5, 9.5)
fp.add_labels(ax_ils[0], 'Constant boundary\ncondition perturbation',
              'Influence length scale', labelpad=10)
fp.add_labels(ax_ils[1], 'Residence time', '', labelpad=10)
# fp.add_labels(ax_ils[1, 0], 'Scale factor applied\nto 'r'x$_{A,0}$',
#               'Influence length scale', labelpad=10)
fp.add_labels(ax_ils[2], 'Scale factor applied\nto 'r'$\sigma_{A,0}$', '', 
              labelpad=10)

# Formatting ax_bc
fp.add_labels(ax_bc[0], r'$\zeta$', r'$\Delta \hat{x}$')
fp.add_labels(ax_bc[1], r'$\zeta$', '')
fp.add_labels(ax_bc[2], r'$\Wind speed', '')
fp.save_fig(fig_ils, plot_dir, f'constant_BC_ILS')

fp.save_fig(fig_summ, plot_dir, f'constant_BC_sv')

# Formatting ax_bc
ax_bc[0].set_yscale('log')
for ax in ax_bc.flatten():
    ax.set_xscale('log')
fp.save_fig(fig_bc, plot_dir, f'constant_BC_fracBC')