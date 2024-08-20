import numpy as np
from scipy.linalg import block_diag
import os
from copy import deepcopy as dc
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.lines import Line2D

# Custom packages
import settings as s
from utilities import plot_settings as ps
from utilities import stats
from utilities import inversion as inv
from utilities import format_plots as fp

rcParams['text.usetex'] = True
np.set_printoptions(precision=2, linewidth=300, suppress=True)

## -------------------------------------------------------------------------##
# File Locations
## -------------------------------------------------------------------------##
plot_dir = f'../plots/n{s.nstate}_m{s.nobs}'
# plot_dir = f'{plot_dir}/n{s.nstate}_m{s.nobs}'
if not os.path.exists(plot_dir):
    os.mkdir(plot_dir)

## -------------------------------------------------------------------------##
# Define sensitivity test parameters
## -------------------------------------------------------------------------##
U = np.concatenate([np.arange(5, 0, -1), 
                    np.arange(1, 5, 1)])*24
U = np.repeat(U, 2)
# U = 5*24

if type(U) in [float, int]:
    suffix = 'constwind'
else:
    suffix = 'varwind'

true_BC = inv.Inversion(U=U, gamma=1)

print(true_BC.estimate_D(10, 0.5))

## -------------------------------------------------------------------------##
# Constant BC perturbations
## -------------------------------------------------------------------------##
fig, ax = fp.get_figax(rows=3, cols=2, aspect=2, max_height=4.5, max_width=10,
                       # width_ratios=[0.25, 1, 1],
                       sharex=True, sharey=True)
ax = ax.T.flatten()
plt.subplots_adjust(wspace=0.7, hspace=0.5)
ax[0].set_ylim(-1.05, 1.05)

# ax0 = ax[0].twinx()
ax1 = ax[1].twinx()
ax1.set_ylim(-0.01, 0.01)
# ax2 = ax[2].twinx()
# ax3 = ax[3].twinx()
# ax4 = ax[4].twinx()
# ax5 = ax[5].twinx()
# axx = [ax0, ax1]#, ax2, ax3, ax4, ax5]

fig_summ, ax_summ = fp.get_figax(cols=3, aspect=1, sharey=True, sharex=False)
ax_summ[0] = fp.add_labels(ax_summ[0], 
                           'Prior error [ppb/hr]', 
                           r'$\Delta\hat{x}/\Delta b$ [hr$^{-1}$]')
ax_summ[1] = fp.add_labels(ax_summ[1], 'Mean grid cell lifetime [hr]', '')
ax_summ[2] = fp.add_labels(ax_summ[2], 
                           'Prior error * mean grid cell lifetime [ppb]', '')
# ppb^2/hr^2
# k = 0.2 hr
# x = ppb/hr
a_std = (true_BC.sa*true_BC.xa_abs**2*0.2**2/
         (true_BC.sa*true_BC.xa_abs**2*0.2**2 + 15**2/true_BC.nobs))
g_std = ((0.2*true_BC.xa_abs*true_BC.sa)/
         ((0.2*true_BC.xa_abs)**2*true_BC.sa + 15**2/true_BC.nobs))

# BC perturbations
# The first axis will plot the effect of BC perturbations on the
# posterior solution
perts = np.arange(5, 55, 10)
for i, pert in enumerate(perts):
    pert_BC = inv.Inversion(
        U=U, gamma=1, BC=true_BC.BCt + pert)

    # Plot the relative error in the posterior solution
    ax[0].plot(
        pert_BC.xp, 
        stats.rel_err(pert_BC.xhat*pert_BC.xa_abs, true_BC.xt_abs),#/pert,
        color=fp.color(8 - 2*i, lut=10), lw=1, ls='-.', label=f'{pert} ppb')

    # Plot the relative error in the posterior solution
    ax[1].plot(
        pert_BC.xp, stats.rel_err(pert_BC.xhat, true_BC.xhat),#/pert,
        color=fp.color(8 - 2*i, lut=10), lw=1, label=f'{pert} ppb')
    ax1.plot(pert_BC.xp,
             pert_BC.estimate_delta_xhat(10)/true_BC.xa_abs,
             color=fp.color(8 - 2*i, lut=10), lw=2, ls='--')
    # axx[1].plot(
    #     pert_BC.xp, pert_BC.bc_contrib/pert_BC.xhat,
    #     color=fp.color(8 - 2*i, lut=10), lw=1, ls='--')
    # # axx[1].plot(
    # #     pert_BC.xp, g_std*pert/true_BC.xhat,
    # #     color=fp.color(8 - 2*i, lut=10), lw=1, ls=':')
    ax_summ[0].scatter(
        pert_BC.sa**0.5*pert_BC.xa_abs,
        stats.rel_err(pert_BC.xhat, true_BC.xhat)/pert*true_BC.xhat,
        c=fp.color(8 - 2*i, lut=10))
    k_temp = pert_BC.k/pert_BC.xa_abs
    k_temp[k_temp == 0] = np.nan
    ax_summ[1].scatter(
        np.nanmean(k_temp, axis=0),
        stats.rel_err(pert_BC.xhat, true_BC.xhat)/pert*true_BC.xhat,
        c=fp.color(8 - 2*i, lut=10))
    ax_summ[2].scatter(
        pert_BC.sa**0.5*pert_BC.xa_abs*np.nanmean(k_temp, axis=0),
        stats.rel_err(pert_BC.xhat, true_BC.xhat)/pert*true_BC.xhat,
        c=fp.color(8 - 2*i, lut=10))

    # delta_signal = (((pert_BC.g - true_BC.g) 
    #                 @ (true_BC.y - true_BC.k @ true_BC.xa - true_BC.c))
    #                 /true_BC.xhat)
    # delta_noise = -(pert_BC.g @ (pert_BC.c - true_BC.c))/true_BC.xhat
    # if suffix == 'constwind':
    #     ax[1].plot(pert_BC.xp, delta_signal,#/pert, 
    #                color=fp.color(8 - 2*i, lut=10), 
    #                lw=2, ls='--')
    #     ax[1].plot(pert_BC.xp, delta_noise,#/pert, 
    #                color=fp.color(8 - 2*i, lut=10), 
    #                lw=2, ls=':')
    # ax[0].plot(
    #     pert_BC.xp, np.abs(stats.rel_err(pert_BC.xhat*pert_BC.xa_abs, 
    #                                      true_BC.xt_abs)),
    #     color=fp.color(8 - 2*i, lut=10), lw=2, ls='--')
    # ax[0].plot(pert_BC.xp, pert*np.abs(pert_BC.g.sum(axis=1))/true_BC.xhat,
    #            color=fp.color(8 - 2*i, lut=10), lw=2, ls='--')

fp.add_title(ax[0], 'Standard inversion compared to truth')
fp.add_legend(ax[0], bbox_to_anchor=(1.2, 0.5), loc='center left', 
              title='BC perturbation', alignment='left')

fp.add_title(ax[1], 'Standard inversion')
fp.add_legend(ax[1], bbox_to_anchor=(1.2, 0.5), loc='center left', 
              title='BC perturbation', alignment='left')

# BC correction
# The third axis will plot the effect of correcting the boundary condition 
# as part of the inversion
# ax2 = ax[2].twinx()
# ax2.set_ylim(-0.05, 0.05)
for i, pert in enumerate(perts):
    pert_opt_BC = inv.Inversion(U=U, gamma=1, BC=true_BC.BCt + pert,
                                opt_BC=True)
    print(true_BC.BCt + pert, pert_opt_BC.xhat_BC)
    
    ax[2].plot(
        pert_opt_BC.xp, stats.rel_err(pert_opt_BC.xhat, true_BC.xhat),#/pert, 
        color=fp.color(8 - 2*i, lut=10), lw=1, label=f'{pert} ppb')
    # axx[2].plot(
    #     pert_opt_BC.xp, pert_opt_BC.bc_contrib/pert_opt_BC.xhat,#/pert, 
    #     color=fp.color(8 - 2*i, lut=10), lw=1, ls='--', label=f'{pert} ppb')

    # ax2.plot(
    #     pert_opt_BC.xp, pert_opt_BC.g.sum(axis=1), 
    #     color=fp.color(8 - 2*i, lut=10), lw=1, label=f'{pert} ppb')
    
    # delta_y_true = true_BC.y - true_BC.k @ true_BC.xa - true_BC.c
    # delta_signal = (((pert_opt_BC.g - true_BC.g) @ delta_y_true)/true_BC.xhat)
    # delta_y_pert = (pert_opt_BC.y - pert_opt_BC.k @ pert_opt_BC.xa)
    # delta_noise = -(pert_opt_BC.g @ (delta_y_pert - delta_y_true))/true_BC.xhat
    # if suffix == 'constwind':
    #     ax[2].plot(pert_opt_BC.xp, delta_signal, color=fp.color(8 - 2*i, lut=10), 
    #             lw=2, ls='--')
    #     ax[2].plot(pert_opt_BC.xp, delta_noise,#/pert, color=fp.color(8 - 2*i, lut=10), 
    #             lw=2, ls=':')

fp.add_title(ax[2], r'Boundary condition correction')
fp.add_legend(ax[2], bbox_to_anchor=(1.2, 0.5), loc='center left', 
              title='BC perturbation', alignment='left')

# Sa_abs scaling
# The second axis will plot the effect of using a buffer grid cell
# ax3 = ax[3].twinx()
# ax3.set_ylim(-0.05, 0.05)
sfs = np.array([1, 5, 10, 50, 100])
for i, sf in enumerate(sfs):
    sa_sf = dc(true_BC.sa)
    sa_sf[0] *= sf**2
    pert_sa_BC = inv.Inversion(
        U=U, sa=sa_sf, BC=true_BC.BCt + pert, gamma=1)

    # Plot the relative error in the posterior solution
    ax[3].plot(
        pert_sa_BC.xp, stats.rel_err(pert_sa_BC.xhat, true_BC.xhat),#/pert, 
        color=fp.color(8 - 2*i, lut=10), lw=1, label=f'{sf}')
    # axx[3].plot(
    #     pert_sa_BC.xp, pert_sa_BC.bc_contrib/pert_sa_BC.xhat,#/pert, 
    #     color=fp.color(8 - 2*i, lut=10), lw=1, ls='--', label=f'{sf}')

    # ax3.plot(
    #     pert_sa_BC.xp, pert_sa_BC.g.sum(axis=1), 
    #     color=fp.color(8 - 2*i, lut=10), lw=1)

    # # Add in optimizing BC 
    # for pert in perts:
    #     pert_opt_BC = inv.Inversion(U=U, gamma=1, BC=true_BC.BCt + pert,
    #                                 opt_BC=True)
    #     ax[3].plot(
    #         pert_opt_BC.xp, 
    #         stats.rel_err(pert_opt_BC.xhat, true_BC.xhat),#/pert, 
    #         color='green', lw=0.5, ls='-.', zorder=20)

    # delta_signal = (((pert_sa_BC.g - true_BC.g) 
    #                 @ (true_BC.y - true_BC.k @ true_BC.xa - true_BC.c))
    #                 /true_BC.xhat)
    # delta_noise = -(pert_sa_BC.g @ (pert_sa_BC.c - true_BC.c))/true_BC.xhat
    # if suffix == 'constwind':
    #     ax[3].plot(pert_sa_BC.xp, delta_signal,#/pert, 
    #                 color=fp.color(8 - 2*i, lut=10), 
    #                 lw=2, ls='--')
    #     ax[3].plot(pert_sa_BC.xp, delta_noise,#/pert, 
    #                 color=fp.color(8 - 2*i, lut=10), 
    #                 lw=2, ls=':')

fp.add_title(ax[3], 'Buffer grid cell\n'f'(perturbation = {pert} ppb)')
fp.add_legend(ax[3], bbox_to_anchor=(1.2, 0.5), loc='center left', 
              title='Prior error scaling\nin buffer grid cell', 
              alignment='left')

# Correction and buffer (Sa_abs scaling)
# The fourth axis will plot the effect of using a buffer grid cell
sfs = np.array([1, 5, 10, 50, 100])
for i, sf in enumerate(sfs):
    sa_sf = dc(true_BC.sa)
    sa_sf[0] *= sf**2
    pert_sa_opt_BC = inv.Inversion(
        U=U, sa=sa_sf, BC=true_BC.BCt + pert, gamma=1, opt_BC=True)

    # Plot the relative error in the posterior solution
    ax[4].plot(
        pert_sa_opt_BC.xp, 
        stats.rel_err(pert_sa_opt_BC.xhat, true_BC.xhat),#/pert, 
        color=fp.color(8 - 2*i, lut=10), lw=1, label=f'{sf}')
    # axx[4].plot(
    #     pert_sa_opt_BC.xp,
    #     pert_sa_opt_BC.bc_contrib/pert_sa_opt_BC.xhat,
    #     color=fp.color(8 - 2*i, lut=10), ls='--', lw=1)
    
    # # Add in optimizing BC 
    # for pert in perts:
    #     pert_opt_BC = inv.Inversion(U=U, gamma=1, BC=true_BC.BCt + pert,
    #                                 opt_BC=True)
    #     ax[4].plot(
    #         pert_opt_BC.xp, 
    #         stats.rel_err(pert_opt_BC.xhat, true_BC.xhat),#/pert, 
    #         color='green', lw=0.5, ls='-.', zorder=20)

    # delta_y_true = true_BC.y - true_BC.k @ true_BC.xa - true_BC.c
    # delta_signal = (((pert_sa_opt_BC.g - true_BC.g) @ delta_y_true)/true_BC.xhat)
    # delta_y_pert = (pert_sa_opt_BC.y - pert_sa_opt_BC.k @ pert_sa_opt_BC.xa)
    # delta_noise = -(pert_sa_opt_BC.g @ (delta_y_pert - delta_y_true))/true_BC.xhat
    # if suffix == 'constwind':
    #     ax[4].plot(pert_sa_opt_BC.xp, delta_signal, color=fp.color(8 - 2*i, lut=10), 
    #                lw=2, ls='--')
    #     ax[4].plot(pert_sa_opt_BC.xp, delta_noise,#/pert, 
    #                color=fp.color(8 - 2*i, lut=10), 
    #                lw=2, ls=':')

fp.add_title(ax[4], 
             'Buffer grid cell and boundary condition\ncorrection '
             f'(perturbation = {pert} ppb)')
fp.add_legend(ax[4], bbox_to_anchor=(1.2, 0.5), loc='center left', 
              title='Prior error scaling\nin buffer grid cell', 
              alignment='left')

# # Sequential update
# # The fourth axis will plot the effect of correcting the boundary condition 
# # as part of the inversion and then doing the inversion
# for i, pert in enumerate(perts):
#     pert_seq_BC = inv.Inversion(U=U, gamma=1, BC=true_BC.BCt + pert,
#                                 opt_BC=True, sequential=True)
    
#     ax[5].plot(
#         pert_seq_BC.xp, stats.rel_err(pert_seq_BC.xhat, true_BC.xhat),#/pert, 
#         color=fp.color(8 - 2*i, lut=10), lw=1, label=f'{pert} ppb')
#     axx[5].plot(
#         pert_seq_BC.xp, pert_seq_BC.bc_contrib/pert_seq_BC.xhat,
#         color=fp.color(8 - 2*i, lut=10), lw=1, ls='--')
    
#     # delta_signal = (((pert_seq_BC.g - true_BC.g) 
#     #                 @ (true_BC.y - true_BC.k @ true_BC.xa - true_BC.c))
#     #                 /true_BC.xhat)
#     # delta_noise = -(pert_seq_BC.g @ (pert_seq_BC.c - true_BC.c))/true_BC.xhat
#     # if suffix == 'constwind':
#     #     ax[5].plot(pert_seq_BC.xp, delta_signal,#/pert, 
#     #                color=fp.color(8 - 2*i, lut=10), 
#     #                lw=2, ls='--')
#     #     ax[5].plot(pert_seq_BC.xp, delta_noise,#/pert, 
#     #                color=fp.color(8 - 2*i, lut=10), 
#     #                lw=2, ls=':')

# fp.add_title(ax[5], r'Sequential update')
# leg5 = ax[5].legend(bbox_to_anchor=(1.2, 0.5), loc='center left', 
#      BC perturbation', alignment='left',
#                     frameon=False, fontsize=ps.LABEL_FONTSIZE*ps.SCALE)

# # Covariance
# # The fifth axis will plot the effect of specifying observing system covariance
# for i, pert in enumerate(perts):
#     so = (np.diag(5*np.ones(s.nstate), k=0) 
#           + np.diag(1*np.ones(s.nstate - 1), k=-1)
#           + np.diag(1*np.ones(s.nstate - 1), k=1))
#     so = block_diag(*[so for j in range(s.nobs_per_cell)])
#     pert_cov_BC = inv.Inversion(U=U, gamma=1, BC=true_BC.BCt + pert,
#                                 so=so)
    
#     ax[5].plot(
#         pert_cov_BC.xp, stats.rel_err(pert_cov_BC.xhat, true_BC.xhat), 
#         color=fp.color(8 - 2*i, lut=10), lw=1, label=f'{pert} ppb')
    
#     delta_signal = (((pert_cov_BC.g - true_BC.g) 
#                     @ (true_BC.y - true_BC.k @ true_BC.xa - true_BC.c))
#                     /true_BC.xhat)
#     delta_noise = -(pert_cov_BC.g @ (pert_cov_BC.c - true_BC.c))/true_BC.xhat
#     ax[5].plot(pert_cov_BC.xp, delta_signal, color=fp.color(8 - 2*i, lut=10), 
#                lw=2, ls='--')
#     ax[5].plot(pert_cov_BC.xp, delta_noise, color=fp.color(8 - 2*i, lut=10), 
#                lw=2, ls=':')

# fp.add_title(ax[5], r'Observing system covariance matrix')
# leg5 = ax[5].legend(bbox_to_anchor=(1.2, 0.5), loc='center left', 
#      BC perturbation', alignment='left',
#                     frameon=False, fontsize=ps.LABEL_FONTSIZE*ps.SCALE)

# Custom legend
# custom_patches = [Line2D([0], [0], lw=1, ls='-.', color='grey'),
#                   Line2D([0], [0], lw=1, ls='-', color='grey'),
#                   Line2D([0], [0], lw=2, ls='--', color='grey'), 
#                   Line2D([0], [0], lw=2, ls=':', color='grey')]
# custom_labels = ['Relative error compared to truth',
#                  r'Change in posterior flux ($\Delta \hat{x}$)',
#                  r'Change in signal ($(G^\prime-G)\Delta y_{emissions}$)',
#                  r'Change in noise ($-G^\prime\Delta B$)']
# ax[5].legend(handles=custom_patches, labels=custom_labels,
#              bbox_to_anchor=(0.5, 0.5), loc='center', ncol=1, frameon=False,
#              bbox_transform=ax[5].transAxes,
#             #  bbox_transform=fig.transFigure, 
#              fontsize=ps.LABEL_FONTSIZE*ps.SCALE)
ax[5].axis('off')

# Re add the fifth legend
# ax[5].add_artist(leg5)

# for i in range(2):
#     axx[i].set_ylim(100, -100)

# General formatting
for i in range(5):
    ax[i].set_xticks(np.arange(0, s.nstate+1, 5))
    ax[i].set_xlim(0.5, s.nstate + 0.5)
    # axx[i].set_ylim(100, -100)
    # ax_a[i].axhline(0, ls='--', lw=1, color='grey')
    # ax_a[i].set_ylim(100.05, 0)
    for k in range(21):
        ax[i].axvline(k + 0.5, c=fp.color(1), alpha=0.2,
                            ls=':', lw=0.5)
    if i in [2, 5]:
        xlabel = 'State vector element'
    else:
        xlabel = ''

    if i in [0, 1, 2]:
        ylabel = r'$\Delta \hat{x}/\Delta \hat{x}_T$'
    else:
        ylabel = ''

    fp.add_labels(ax[i], xlabel, ylabel)

fp.save_fig(fig, plot_dir, f'constant_BC_{suffix}')

fp.save_fig(fig_summ, plot_dir, f'params_BC_{suffix}')