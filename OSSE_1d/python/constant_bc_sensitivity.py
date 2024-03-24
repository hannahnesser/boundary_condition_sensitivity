import numpy as np
import os
from copy import deepcopy as dc
import matplotlib.pyplot as plt
from matplotlib import rcParams

# Custom packages
import settings as s
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
# # Default is opt_BC = False
# U = np.concatenate([np.arange(5, 0, -1), 
#                     np.array([0.1, -0.1]), 
#                     np.arange(-1, -5, -1)])*24
# U = np.repeat(U, 2)
U = 5*24

opt_BC = False
suffix = 'nooptBC'

true_BC = inv.Inversion(U=U, opt_BC=opt_BC, gamma=1)

# Print out a 2 x 2 example with
# (intereted in whatever is applied to delta BC--so maybe just
# print xhat - xhat_true)
# - standard
# - BC correction
# - Buffer grid cell
# - Combination

# sa_sf = dc(true_BC.sa)
# sa_sf[0] *= 5**2
# pert_sa = inv.Inversion(U=U, sa=sa_sf, gamma=1 opt_BC=opt_BC)
# print(pert_sa.g.sum(axis=1))

# pert_sa_bc = inv.Inversion(U=U, sa=sa_sf, gamma=1 BC=1910, opt_BC=opt_BC)
# print((pert_sa_bc.g @ (10*np.ones((pert_sa_bc.g.shape[1], 1)))).reshape(-1,))

## -------------------------------------------------------------------------##
# Constant BC perturbations
## -------------------------------------------------------------------------##
fig, ax = fp.get_figax(rows=3, cols=2, aspect=2, max_height=4.5, max_width=5,
                       # width_ratios=[0.25, 1, 1],
                       sharex=True, sharey='col')
ax = ax.flatten()
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
        U=U, opt_BC=opt_BC, gamma=1, BC=true_BC.BCt + pert)

    # Plot the relative error in the posterior solution
    ax[0].plot(
        pert_BC.xp, stats.rel_err(pert_BC.xhat, true_BC.xhat),
        color=fp.color(8 - 2*i, lut=10), lw=1, label=f'{pert} ppb')
    
    delta_signal = (((pert_BC.g - true_BC.g) 
                    @ (true_BC.y - true_BC.k @ true_BC.xa - true_BC.c))
                    /true_BC.xhat)
    delta_noise = -pert*pert_BC.g.sum(axis=1)/true_BC.xhat
    ax[0].plot(pert_BC.xp, delta_signal, color=fp.color(8 - 2*i, lut=10), 
               lw=2, ls='--')
    ax[0].plot(pert_BC.xp, delta_noise, color=fp.color(8 - 2*i, lut=10), 
               lw=2, ls=':')
    # ax[0].plot(
    #     pert_BC.xp, np.abs(stats.rel_err(pert_BC.xhat*pert_BC.xa_abs, 
    #                                      true_BC.xt_abs)),
    #     color=fp.color(8 - 2*i, lut=10), lw=2, ls='--')
    # ax[0].plot(pert_BC.xp, pert*np.abs(pert_BC.g.sum(axis=1))/true_BC.xhat,
    #            color=fp.color(8 - 2*i, lut=10), lw=2, ls='--')

fp.add_title(ax[0], 'Standard boundary condition perturbations')
fp.add_legend(ax[0], bbox_to_anchor=(1.15, 0.5), loc='center left')

# Sa_abs scaling
# The second axis will plot the effect of using a buffer grid cell
sfs = np.array([1, 5, 10, 15, 20])
for i, sf in enumerate(sfs):
    sa_sf = dc(true_BC.sa)
    if opt_BC:
        sa_sf = sa_sf[:-1]
    sa_sf[0] *= sf**2
    pert_sa_BC = inv.Inversion(
        U=U, sa=sa_sf, BC=true_BC.BCt + pert, gamma=1, opt_BC=opt_BC)

    # Plot the relative error in the posterior solution
    ax[1].plot(
        pert_sa_BC.xp, stats.rel_err(pert_sa_BC.xhat, true_BC.xhat), 
        color=fp.color(8 - 2*i, lut=10), lw=1, label=f'{sf}')
    
    delta_signal = (((pert_sa_BC.g - true_BC.g) 
                    @ (true_BC.y - true_BC.k @ true_BC.xa - true_BC.c))
                    /true_BC.xhat)
    delta_noise = -pert*pert_sa_BC.g.sum(axis=1)/true_BC.xhat
    ax[1].plot(pert_sa_BC.xp, delta_signal, color=fp.color(8 - 2*i, lut=10), 
               lw=2, ls='--')
    ax[1].plot(pert_sa_BC.xp, delta_noise, color=fp.color(8 - 2*i, lut=10), 
               lw=2, ls=':')

    # ax[1].plot(
    #     pert_sa_BC.xp, np.abs(stats.rel_err(pert_sa_BC.xhat*pert_sa_BC.xa_abs,
    #                                         true_BC.xt_abs)), 
    #     color=fp.color(8 - 2*i, lut=10), lw=2, ls='--')


fp.add_title(ax[1], r'Buffer grid cell')
fp.add_legend(ax[1], bbox_to_anchor=(1.15, 0.5), loc='center left')

# # Larger buffer grid cell sa_abs scaling
# # The third axis will plot the effect of aggregating together grid 
# # boxes to create a larger buffer grid cell
# nbuff = 2
# for i, sf in enumerate(sfs):
#     # Get all the components of the inversion 
#     ## Sa
#     sa_sf = dc(true_BC.sa[(nbuff - 1):])
#     sa_sf[0] *= sf**2
#     sa_inv = np.diag(1/sa_sf)
#     print('-'*70)
#     print(sf)
#     print(sa_sf)

#     ## So
#     so = dc(true_BC.so)
#     gamma = 1
#     so_inv = np.diag(gamma/so)

#     ## xa and y and c
#     xa = dc(pert_BC.xa)[(nbuff - 1):]
#     xa_abs = dc(pert_BC.xa_abs)
#     xa_abs = np.concatenate([xa_abs[:nbuff].sum().reshape((-1, 1)), 
#                              xa_abs[nbuff:].reshape((-1, 1))])
#     y = dc(pert_BC.y)
#     c = dc(pert_BC.c)

#     ## K
#     kfull = dc(pert_BC.k)
#     k = np.concatenate([kfull[:, :nbuff].sum(axis=1).reshape((-1, 1)),
#                         kfull[:, nbuff:]], axis=1)

#     # Solve the inversion
#     shat = np.linalg.inv(sa_inv + k.T @ so_inv @ k)
#     g = (shat @ k.T @ so_inv)
#     xhat = (xa + g @ (y - k @ xa - c))
#     a = (g @ k)

#     # Get the boundary condition contribution
#     xa_contrib = a @ xa
#     bc_contrib = g @ c
#     tot_correct = g @ y - (bc_contrib + xa_contrib)

#     # Filter
#     if opt_BC:
#         xhat = xhat[:-1]
#         shat = shat[:-1, :-1]
#         a = a[:-1, :-1]
#         g = g[:-1, :]
#         bc_contrib = bc_contrib[:-1]
#         xa_contrib = xa_contrib[:-1]
#         tot_correct = tot_correct[:-1]

#     # Get xp for plotting
#     xp = np.concatenate([[(1 + nbuff)/2], 
#                           np.arange(nbuff + 1, true_BC.nstate + 1)])

#     ax[2].plot(
#         xp, np.abs(stats.rel_err(xhat, true_BC.xhat[(nbuff - 1):])),
#         color=fp.color(8 - 2*i, lut=10), lw=2, label=f'{sf}')
    
#     delta_signal = (((g - true_BC.g) 
#                     @ (true_BC.y - true_BC.k @ true_BC.xa - true_BC.c))
#                     /true_BC.xhat)
#     delta_noise = -pert*pert_sa_BC.g.sum(axis=1)/true_BC.xhat
#     ax[1].plot(pert_sa_BC.xp, delta_signal, color=fp.color(8 - 2*i, lut=10), 
#                lw=2, ls='--')
#     ax[1].plot(pert_sa_BC.xp, delta_noise, color=fp.color(8 - 2*i, lut=10), 
#                lw=2, ls=':')

#     # ax[2].plot(
#     #     xp, np.abs(stats.rel_err(xhat*xa_abs, 
#     #                              true_BC.xt_abs[(nbuff - 1):])),
#     #     color=fp.color(8 - 2*i, lut=10), lw=2, ls='--')
#     # ax_a[-1].plot(xp, np.abs(bc_contrib/xhat),
#     #                   color=fp.color(8 - 2*i, lut=10), ls='--', lw=1)
#     # ax_a[-1].plot(xp, np.abs(g.sum(axis=1)),
#     #              color=fp.color(8 - 2*i, lut=10), ls=':', lw=1)
#     # ax_b[-1].plot(xp, g @ y,#/tot_correct),
#     #                   color=fp.color(8 - 2*i, lut=10), ls=':', lw=1)
#     # ax_c[-1].plot(true_BC.xp, true_BC.xa_abs,#/np.abs(bc_contrib),
#     #               color='black', ls='-', lw=1)

# fp.add_title(ax[2], r'Scaling $\sigma_{A,0}$ with larger buffer grid cell')
# fp.add_legend(ax[2], bbox_to_anchor=(1.15, 0.5), loc='center left')

# True BC
# The third axis will plot the effect of correcting the boundary condition 
# as part of the inversion
perts = np.arange(1, 10, 2)
for i, pert in enumerate(perts):
    pert_opt_BC = inv.Inversion(U=U, gamma=1, BC=true_BC.BCt + pert,
                                opt_BC=True)
    
    ax[-1].plot(
        pert_opt_BC.xp, stats.rel_err(pert_opt_BC.xhat, true_BC.xhat), 
        color=fp.color(8 - 2*i, lut=10), lw=1, label=f'{pert}')
    
    delta_signal = (((pert_opt_BC.g - true_BC.g) 
                    @ (true_BC.y - true_BC.k @ true_BC.xa - true_BC.c))
                    /true_BC.xhat)
    delta_noise = -pert*pert_opt_BC.g.sum(axis=1)/true_BC.xhat
    ax[-1].plot(pert_opt_BC.xp, delta_signal, color=fp.color(8 - 2*i, lut=10), 
               lw=2, ls='--')
    ax[-1].plot(pert_opt_BC.xp, delta_noise, color=fp.color(8 - 2*i, lut=10), 
               lw=2, ls=':')

fp.add_title(ax[-1], r'Boundary condition correction')
fp.add_legend(ax[-1], bbox_to_anchor=(1.15, 0.5), loc='center left')

# General formatting
for i in range(3):
    ax[i].set_xticks(np.arange(0, s.nstate+1, 5))
    ax[i].set_xlim(0.5, s.nstate + 0.5)
    # ax_a[i].axhline(0, ls='--', lw=1, color='grey')
    # ax_a[i].set_ylim(100.05, 0)
    for k in range(21):
        ax[i].axvline(k + 0.5, c=fp.color(1), alpha=0.2,
                            ls=':', lw=0.5)
    if i == 2:
        xlabel = 'State vector element'
    else:
        xlabel = ''

    fp.add_labels(ax[i], xlabel, r'$\Delta x$')
    # fp.add_labels(ax_a[i], '', r'$\Sigma$ G')
    # fp.add_labels(ax_a[i], '', r'$\zeta$')
    # fp.add_labels(ax_a[i], '', r'$\vert Gc/\hat{x} \vert$')
    # fp.add_labels(ax_b[i], '', r'$Gy$')
    # fp.add_labels(ax_c[i], '', r'$x_A$')


fp.save_fig(fig, plot_dir, f'constant_BC_{suffix}')