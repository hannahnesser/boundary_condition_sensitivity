import numpy as np
import pandas as pd
from copy import deepcopy as dc
import os
import matplotlib.pyplot as plt

# Custom packages
import sys
sys.path.append('.')
import settings as s
import gcpy as gc
import inversion as inv
import plot_settings as ps
ps.SCALE = ps.PRES_SCALE
ps.BASE_WIDTH = ps.PRES_WIDTH
ps.BASE_HEIGHT = ps.PRES_HEIGHT
import format_plots as fp

np.set_printoptions(precision=2, linewidth=300, suppress=True)

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
# # Default is opt_BC = False
# U = np.concatenate([np.arange(5, 0, -1), 
#                     np.array([0.1, -0.1]), 
#                     np.arange(-1, -5, -1)])*24
# U = np.repeat(U, 2)
U = 5*24

opt_BC = True

suffix = 'constU_optBC'

true_BC = inv.Inversion(U=U, opt_BC=opt_BC)

# ## -------------------------------------------------------------------------##
# # ILS sensitivity perturbations
# ## -------------------------------------------------------------------------##
# fig, ax = fp.get_figax(rows=3, cols=1, aspect=3,
#                        max_height=4.5, max_width=5,
#                        # width_ratios=[0.25, 1, 1],
#                        sharex=True, sharey='col')
# plt.subplots_adjust(wspace=0.5, hspace=0.5)
# ax[0].set_ylim(0, 1.05)

# # BC perturbations
# # The first axis will plot the effect of BC perturbations on the
# # posterior solution without any modifications
# perts = np.arange(5, 55, 10)
# for i, pert in enumerate(perts):
#     # First, plot the relative error in the posterior solution
#     pert_BC = inv.Inversion(U=U, BC=true_BC.BC_t + pert, #gamma=1,
#                             opt_BC=opt_BC)
#     ax[0].plot(pert_BC.xp, np.abs(gc.rel_err(pert_BC.xhat, true_BC.xhat)),
#                color=fp.color(2*i, lut=10), lw=2,
#                label=f'{pert} ppb')

# fp.add_title(ax[0], 'Boundary condition perturbations')
# fp.add_legend(ax[0], bbox_to_anchor=(1.15, 0.5), loc='center left')

# # Sa_abs scaling
# # The second axis will plot the effect of using a buffer grid cell
# sfs = np.array([1, 2, 3, 4, 5])
# for i, sf in enumerate(sfs):
#     sa_sf = dc(true_BC.sa)
#     if opt_BC:
#         sa_sf = sa_sf[:-1]
#     sa_sf[0] *= sf**2
#     pert_sa_BC = inv.Inversion(U=U, sa=sa_sf, #gamma=pert_BC.gamma,
#                                BC=true_BC.BC_t + pert,
#                                opt_BC=opt_BC)
#     true_sa_BC = inv.Inversion(U=U, sa=sa_sf, #gamma=pert_BC.gamma,
#                                BC=true_BC.BC_t,
#                                opt_BC=opt_BC)
#     ax[1].plot(
#         pert_sa_BC.xp, np.abs(gc.rel_err(pert_sa_BC.xhat, true_BC.xhat)), 
#         color=fp.color(2*i, lut=10), lw=2, label=f'{sf}')

# fp.add_title(ax[1], r'Scaling $\sigma_{A,0}$')
# fp.add_legend(ax[1], bbox_to_anchor=(1.15, 0.5), loc='center left')

# # Larger buffer grid cell sa_abs scaling
# # The third axis will plot the effect of aggregating together grid 
# # boxes to create a larger buffer grid cell
# nbuff = 2
# for i, sf in enumerate(sfs):
#     # Get all the components of the inversion 
#     ## xa and y and c
#     xa = dc(pert_BC.xa)[(nbuff - 1):]
#     xa_abs = dc(pert_BC.xa_abs)
#     print(xa_abs)
#     xa_abs = np.concatenate([xa_abs[:nbuff].sum().reshape(-1,), 
#                              xa_abs[nbuff:].reshape(-1,)])
#     y = dc(pert_BC.y)
#     c = dc(pert_BC.c)

#     ## K
#     kfull = dc(pert_BC.k)
#     k = np.concatenate([kfull[:, :nbuff].sum(axis=1).reshape((-1, 1)),
#                         kfull[:, nbuff:]], axis=1)

#     ## Sa
#     sa_sf = dc(true_BC.sa[(nbuff - 1):])
#     sa_sf[0] *= sf**2
#     sa_inv = np.diag(1/sa_sf)

#     ## So
#     so = dc(true_BC.so)
#     gamma = 10
#     gamma_not_found = True
#     while gamma_not_found:
#         so_new_inv = np.diag(so/gamma)
#         shat = np.linalg.inv(sa_inv + k.T @ so_new_inv @ k)
#         g = (shat @ k.T @ so_new_inv)
#         xhat = (xa + g @ (y - k @ xa - c))
#         cost = ((xhat - xa)**2/sa_sf).sum()/len(sa_sf)
#         if np.abs(cost - 1) <= 1e-1:
#             gamma_not_found = False
#         elif cost > 1:
#             gamma /= 2
#         elif cost <1:
#             gamma *= 1.5
#     so_inv = np.diag(gamma/so)

#     fig1, ax1 = fp.get_figax(aspect=3)
#     ysim_full = (kfull @ pert_BC.xa + pert_BC.c).reshape((s.nstate, -1))
#     ysim = (k @ xa + c).reshape((s.nstate, -1))
#     for t in range(ysim.shape[1]):
#         ax1.plot(ysim_full[:, t], color='blue', marker='^')
#         ax1.plot(ysim[:, t], color='red', ls='--', marker='o', lw=0.5)
#     plt.show()

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

#     ax[2].plot(xp, 
#                 np.abs(xhat - true_BC.xhat[(nbuff - 1):])/true_BC.xhat[(nbuff - 1):], 
#                 color=fp.color(2*i, lut=10), lw=2, label=f'{sf}')

# fp.add_title(ax[2], r'Scaling $\sigma_{A,0}$ with larger buffer grid cell')
# fp.add_legend(ax[2], bbox_to_anchor=(1.15, 0.5), loc='center left')

# # # So_abs scaling
# # # The fourth axis will plot the effect of increasing obs errors over 
# # # the first grid cell
# # sfs = np.array([1, 2, 3, 4, 5])
# # for i, sf in enumerate(sfs):
# #     so_sf = dc(true_BC.so)
# #     so_sf[:s.nobs_per_cell] *= sf**2
# #     pert_so_BC = inv.Inversion(U=U, so=so_sf, gamma=pert_BC.gamma,
# #                                BC=true_BC.BC_t + pert,
# #                                opt_BC=opt_BC)
# #     ax[-1].plot(pert_so_BC.xp, 
# #                np.abs(pert_so_BC.xhat - true_BC.xhat)/true_BC.xhat, 
# #                color=fp.color(2*i, lut=10), lw=2, label=f'{sf}')
# #     ax_a[-1].plot(pert_BC.xp, np.abs(pert_so_BC.bc_contrib/pert_so_BC.xhat),
# #                  color=fp.color(2*i, lut=10), ls=':', lw=1)

# # fp.add_title(ax[-1], r'Scaling $\sigma_{O,0}$')
# # fp.add_legend(ax[-1], bbox_to_anchor=(1.15, 0.5), loc='center left')

# # General formatting
# for i in range(3):
#     ax[i].set_xticks(np.arange(0, s.nstate+1, 5))
#     ax[i].set_xlim(0.5, s.nstate + 0.5)
#     # ax_a[i].axhline(0, ls='--', lw=1, color='grey')
#     # ax_a[i].set_ylim(100.05, 0)
#     for k in range(21):
#         ax[i].axvline(k + 0.5, c=fp.color(1), alpha=0.2,
#                             ls=':', lw=0.5)
#     if i == 2:
#         xlabel = 'State vector element'
#     else:
#         xlabel = ''

#     fp.add_labels(ax[i], xlabel, r'$\Delta x$')

# fp.save_fig(fig, plot_dir, f'constant_BC_{suffix}')