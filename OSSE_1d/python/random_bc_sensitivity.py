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
# U = np.concatenate([np.arange(5, 0, -1), 
#                     np.array([0.1, -0.1]), 
#                     np.arange(-1, -5, -1)])*24
# U = np.concatenate([U, U[::-1]])
U = 5*24

if type(U) in [float, int]:
    suffix = 'constwind'
else:
    suffix = 'varwind'

true_BC = inv.Inversion(U=U, gamma=1)

pert_random = np.random.RandomState(s.random_state).normal(
    0, 5, (len(true_BC.t,)))

# # Print out a 2 x 2 example with
# # (intereted in whatever is applied to delta BC--so maybe just
# # print xhat - xhat_true)
# # - standard
# # - BC correction
# # - Buffer grid cell
# # - Combination

# # sa_sf = dc(true_BC.sa)
# # sa_sf[0] *= 5**2
# # pert_sa = inv.Inversion(U=U, sa=sa_sf, gamma=1)
# # print(pert_sa.g.sum(axis=1))

# # pert_sa_bc = inv.Inversion(U=U, sa=sa_sf, gamma=1, BC=1910)
# # print((pert_sa_bc.g @ (10*np.ones((pert_sa_bc.g.shape[1], 1)))).reshape(-1,))

## -------------------------------------------------------------------------##
# Constant BC perturbations
## -------------------------------------------------------------------------##
fig, ax = fp.get_figax(rows=3, cols=2, aspect=2, max_height=4.5, max_width=10,
                       # width_ratios=[0.25, 1, 1],
                       sharex=True, sharey=True)
ax = ax.T.flatten()
plt.subplots_adjust(wspace=0.5, hspace=0.5)
ax[0].set_ylim(-1.05, 1.05)
# ax_a = [ax[i].twinx() for i in range(3)]
# ax_b = [ax[i].twinx() for i in range(3)]
# ax_c = [ax[i].twinx() for i in range(3)]

# BC perturbations
# The first axis will plot the effect of BC perturbations on the
# posterior solution
perts = np.arange(5, 55, 10)
for i, pert in enumerate(perts):
    pert_BC = inv.Inversion(
        U=U, gamma=1, BC=true_BC.BCt + pert_random + pert)

    # Plot the relative error in the posterior solution
    ax[0].plot(
        pert_BC.xp, stats.rel_err(pert_BC.xhat*pert_BC.xa_abs, true_BC.xt_abs),
        color=fp.color(8 - 2*i, lut=10), lw=1, ls='-.', label=f'{pert} ppb')

    # Plot the relative error in the posterior solution
    ax[1].plot(
        pert_BC.xp, stats.rel_err(pert_BC.xhat, true_BC.xhat),
        color=fp.color(8 - 2*i, lut=10), lw=1, label=f'{pert} ppb')
    
    delta_signal = (((pert_BC.g - true_BC.g) 
                    @ (true_BC.y - true_BC.k @ true_BC.xa - true_BC.c))
                    /true_BC.xhat)
    delta_noise = -(pert_BC.g @ (pert_BC.c - true_BC.c))/true_BC.xhat
    ax[1].plot(pert_BC.xp, delta_signal, color=fp.color(8 - 2*i, lut=10), 
               lw=2, ls='--')
    ax[1].plot(pert_BC.xp, delta_noise, color=fp.color(8 - 2*i, lut=10), 
               lw=2, ls=':')
    # ax[0].plot(
    #     pert_BC.xp, np.abs(stats.rel_err(pert_BC.xhat*pert_BC.xa_abs, 
    #                                      true_BC.xt_abs)),
    #     color=fp.color(8 - 2*i, lut=10), lw=2, ls='--')
    # ax[0].plot(pert_BC.xp, pert*np.abs(pert_BC.g.sum(axis=1))/true_BC.xhat,
    #            color=fp.color(8 - 2*i, lut=10), lw=2, ls='--')

fp.add_title(ax[0], 'Standard inversion compared to truth')
fp.add_legend(ax[0], bbox_to_anchor=(1.05, 0.5), loc='center left', 
              title='Boundary condition\nperturbation', alignment='left')

fp.add_title(ax[1], 'Standard inversion')
fp.add_legend(ax[1], bbox_to_anchor=(1.05, 0.5), loc='center left', 
              title='Boundary condition\nperturbation', alignment='left')

# BC correction
# The third axis will plot the effect of correcting the boundary condition 
# as part of the inversion
# ax2 = ax[2].twinx()
# ax2.set_ylim(-0.05, 0.05)
for i, pert in enumerate(perts):
    pert_opt_BC = inv.Inversion(U=U, gamma=1, BC=true_BC.BCt + pert_random + pert,
                                opt_BC=True)
    
    ax[2].plot(
        pert_opt_BC.xp, stats.rel_err(pert_opt_BC.xhat, true_BC.xhat), 
        color=fp.color(8 - 2*i, lut=10), lw=1, label=f'{pert} ppb')

    # ax2.plot(
    #     pert_opt_BC.xp, pert_opt_BC.g.sum(axis=1), 
    #     color=fp.color(8 - 2*i, lut=10), lw=1, label=f'{pert} ppb')
    
    delta_y_true = true_BC.y - true_BC.k @ true_BC.xa - true_BC.c
    delta_signal = (((pert_opt_BC.g - true_BC.g) @ delta_y_true)/true_BC.xhat)
    delta_y_pert = (pert_opt_BC.y - pert_opt_BC.k @ pert_opt_BC.xa)
    delta_noise = -(pert_opt_BC.g @ (delta_y_pert - delta_y_true))/true_BC.xhat
    ax[2].plot(pert_opt_BC.xp, delta_signal, color=fp.color(8 - 2*i, lut=10), 
               lw=2, ls='--')
    if suffix == 'constwind':
        ax[2].plot(pert_opt_BC.xp, delta_noise, color=fp.color(8 - 2*i, lut=10), 
                lw=2, ls=':')

fp.add_title(ax[2], r'Boundary condition correction')
fp.add_legend(ax[2], bbox_to_anchor=(1.05, 0.5), loc='center left', 
              title='Boundary condition\nperturbation', alignment='left')

# Sa_abs scaling
# The second axis will plot the effect of using a buffer grid cell
# ax3 = ax[3].twinx()
# ax3.set_ylim(-0.05, 0.05)
sfs = np.array([1, 5, 10, 50, 100])
for i, sf in enumerate(sfs):
    sa_sf = dc(true_BC.sa)
    sa_sf[0] *= sf**2
    pert_sa_BC = inv.Inversion(
        U=U, sa=sa_sf, BC=true_BC.BCt + pert_random + pert, gamma=1)

    # Plot the relative error in the posterior solution
    ax[3].plot(
        pert_sa_BC.xp, stats.rel_err(pert_sa_BC.xhat, true_BC.xhat), 
        color=fp.color(8 - 2*i, lut=10), lw=1, label=f'{sf}')

    # ax3.plot(
    #     pert_sa_BC.xp, pert_sa_BC.g.sum(axis=1), 
    #     color=fp.color(8 - 2*i, lut=10), lw=1)

    # Add in optimizing BC 
    for pert in perts:
        pert_opt_BC = inv.Inversion(U=U, gamma=1, BC=true_BC.BCt + pert_random + pert,
                                    opt_BC=True)
        ax[3].plot(
            pert_opt_BC.xp, stats.rel_err(pert_opt_BC.xhat, true_BC.xhat), 
            color='green', lw=0.5, ls='-.', zorder=20)

    
    delta_signal = (((pert_sa_BC.g - true_BC.g) 
                    @ (true_BC.y - true_BC.k @ true_BC.xa - true_BC.c))
                    /true_BC.xhat)
    delta_noise = -(pert_sa_BC.g @ (pert_sa_BC.c - true_BC.c))/true_BC.xhat
    ax[3].plot(pert_sa_BC.xp, delta_signal, color=fp.color(8 - 2*i, lut=10), 
               lw=2, ls='--')
    ax[3].plot(pert_sa_BC.xp, delta_noise, color=fp.color(8 - 2*i, lut=10), 
            lw=2, ls=':')

fp.add_title(ax[3], 'Buffer grid cell\n'f'(perturbation = {pert} ppb)')
fp.add_legend(ax[3], bbox_to_anchor=(1.05, 0.5), loc='center left', 
              title='Prior error scaling\nin buffer grid cell', 
              alignment='left')

# Correction and buffer (Sa_abs scaling)
# The fourth axis will plot the effect of using a buffer grid cell
sfs = np.array([1, 5, 10, 50, 100])
for i, sf in enumerate(sfs):
    sa_sf = dc(true_BC.sa)
    sa_sf[0] *= sf**2
    pert_sa_opt_BC = inv.Inversion(
        U=U, sa=sa_sf, BC=true_BC.BCt + pert_random + pert, gamma=1, opt_BC=True)

    # Plot the relative error in the posterior solution
    ax[4].plot(
        pert_sa_opt_BC.xp, stats.rel_err(pert_sa_opt_BC.xhat, true_BC.xhat), 
        color=fp.color(8 - 2*i, lut=10), lw=1, label=f'{sf}')
    
    # Add in optimizing BC 
    for pert in perts:
        pert_opt_BC = inv.Inversion(U=U, gamma=1, BC=true_BC.BCt + pert_random + pert,
                                    opt_BC=True)
        ax[4].plot(
            pert_opt_BC.xp, stats.rel_err(pert_opt_BC.xhat, true_BC.xhat), 
            color='green', lw=0.5, ls='-.', zorder=20)

    
    delta_y_true = true_BC.y - true_BC.k @ true_BC.xa - true_BC.c
    delta_signal = (((pert_sa_opt_BC.g - true_BC.g) @ delta_y_true)/true_BC.xhat)
    delta_y_pert = (pert_sa_opt_BC.y - pert_sa_opt_BC.k @ pert_sa_opt_BC.xa)
    delta_noise = -(pert_sa_opt_BC.g @ (delta_y_pert - delta_y_true))/true_BC.xhat
    ax[4].plot(pert_sa_opt_BC.xp, delta_signal, color=fp.color(8 - 2*i, lut=10), 
               lw=2, ls='--')
    if suffix == 'constwind':
        ax[4].plot(pert_sa_opt_BC.xp, delta_noise, color=fp.color(8 - 2*i, lut=10), 
                   lw=2, ls=':')

fp.add_title(ax[4], 'Buffer grid cell and boundary condition\ncorrection 'f'(perturbation = {pert} ppb)')
fp.add_legend(ax[4], bbox_to_anchor=(1.05, 0.5), loc='center left', 
              title='Prior error scaling\nin buffer grid cell', 
              alignment='left')

# Sequential update
# The fourth axis will plot the effect of correcting the boundary condition 
# as part of the inversion and then doing the inversion
for i, pert in enumerate(perts):
    pert_seq_BC = inv.Inversion(U=U, gamma=1, BC=true_BC.BCt + pert_random + pert,
                                opt_BC=True, sequential=True)
    
    ax[5].plot(
        pert_seq_BC.xp, stats.rel_err(pert_seq_BC.xhat, true_BC.xhat), 
        color=fp.color(8 - 2*i, lut=10), lw=1, label=f'{pert} ppb')
    
    delta_signal = (((pert_seq_BC.g - true_BC.g) 
                    @ (true_BC.y - true_BC.k @ true_BC.xa - true_BC.c))
                    /true_BC.xhat)
    delta_noise = -(pert_seq_BC.g @ (pert_seq_BC.c - true_BC.c))/true_BC.xhat
    ax[5].plot(pert_seq_BC.xp, delta_signal, color=fp.color(8 - 2*i, lut=10), 
               lw=2, ls='--')
    ax[5].plot(pert_seq_BC.xp, delta_noise, color=fp.color(8 - 2*i, lut=10), 
               lw=2, ls=':')

fp.add_title(ax[5], r'Sequential update')
leg5 = ax[5].legend(bbox_to_anchor=(1.05, 0.5), loc='center left', 
                    title='Boundary condition\nperturbation', alignment='left',
                    frameon=False, fontsize=ps.LABEL_FONTSIZE*ps.SCALE)

# # Covariance
# # The fifth axis will plot the effect of specifying observing system covariance
# for i, pert in enumerate(perts):
#     so = (np.diag(5*np.ones(s.nstate), k=0) 
#           + np.diag(1*np.ones(s.nstate - 1), k=-1)
#           + np.diag(1*np.ones(s.nstate - 1), k=1))
#     so = block_diag(*[so for j in range(s.nobs_per_cell)])
#     pert_cov_BC = inv.Inversion(U=U, gamma=1, BC=true_BC.BCt + pert_random + pert,
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
# leg5 = ax[5].legend(bbox_to_anchor=(1.05, 0.5), loc='center left', 
#                     title='Boundary condition\nperturbation', alignment='left',
#                     frameon=False, fontsize=ps.LABEL_FONTSIZE*ps.SCALE)

# Custom legend
custom_patches = [Line2D([0], [0], lw=1, ls='-.', color='grey'),
                  Line2D([0], [0], lw=1, ls='-', color='grey'),
                  Line2D([0], [0], lw=2, ls='--', color='grey'), 
                  Line2D([0], [0], lw=2, ls=':', color='grey')]
custom_labels = ['Relative error compared to truth',
                 r'Change in posterior flux ($\Delta \hat{x}$)',
                 r'Change in signal ($(G^\prime-G)\Delta y_{emissions}$)',
                 r'Change in noise ($-G^\prime\Delta B$)']
ax[5].legend(handles=custom_patches, labels=custom_labels,
             bbox_to_anchor=(0.5, 0), loc='center', ncol=2,
             bbox_transform=fig.transFigure, frameon=False,
             fontsize=ps.LABEL_FONTSIZE*ps.SCALE)

# Re add the fifth legend
ax[5].add_artist(leg5)

# General formatting
for i in range(5):
    ax[i].set_xticks(np.arange(0, s.nstate+1, 5))
    ax[i].set_xlim(0.5, s.nstate + 0.5)
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
        ylabel = r'$\Delta x$'
    else:
        ylabel = ''

    fp.add_labels(ax[i], xlabel, ylabel)

fp.save_fig(fig, plot_dir, f'random_BC_{suffix}')