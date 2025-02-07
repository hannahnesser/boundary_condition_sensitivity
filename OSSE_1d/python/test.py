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

# rcParams['text.usetex'] = True
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
#                     np.arange(1, 5, 1)])*24
# U = np.repeat(U, 2)
U = 5*24

if type(U) in [float, int]:
    suffix = 'constwind'
else:
    suffix = 'varwind'

true_BC = inv.Inversion(U=U, gamma=1)
print(f'True Kx : {(true_BC.k @ true_BC.xa)[:5]}')
kx = true_BC.k @ true_BC.xa
print(f'True Kx mean : {kx.mean()}')
# zeta = true_BC.estimate_delta_xhat(sa_bc=10)
# fig, ax = fp.get_figax() 
# ax.plot(cov_xD, label='xD variance', color=fp.color(2), ls='-.')
# ax.plot(cov_x, label='x variance', color=fp.color(4), ls=':')
# ax.plot(cov_xxD, label='x-xD covariance', color=fp.color(6), ls='--')
# ax.legend(frameon=False, bbox_to_anchor=(1, 0.5), loc='center left')
# ax.set_xticks(np.arange(0, 20))
# ax.set_xticklabels(np.arange(1, 21))
# ax.set_xlim(-0.5, 19.5)
# ax.set_xlabel('State vector element')
# ax.set_ylabel('Posterior\ncovariance\n[relative]')
# fp.save_fig(fig, plot_dir, f'predicted_cov')

# print(true_BC.estimate_D(10, 0.25))

## -------------------------------------------------------------------------##
# Constant BC perturbations
## -------------------------------------------------------------------------##
fig, ax = fp.get_figax(rows=2, cols=3, aspect=2, max_height=4.5, max_width=10,
                       # width_ratios=[0.25, 1, 1],
                       sharex=True, sharey=True)
ax = ax.T.flatten()
plt.subplots_adjust(hspace=0.7)
ax[0].set_ylim(-1.05, 1.05)

perts = [5] #np.arange(5, 55, 10)

# fp.add_title(ax[0], 
#              'Perturbed posterior emissions\ncompared to true emissions')
fp.add_title(ax[1], 
             'Perturbed posterior emissions\n'
             'compared to true posterior emissions')
fp.add_title(ax[2], 
             'Previewed error metric')
fp.add_title(ax[3], 
             'Diagnostic error metric')
fp.add_title(ax[4], 
             'Perturbed posterior emissions with\n'
             'correction method\n'
             'compared to true posterior emissions')
fp.add_title(ax[5], 
             'Perturbed posterior emissions with\n'
             'buffer method\n'
             'compared to true posterior emissions\n'
             f'(perturbation = {perts[-1]} ppb)')


# BC perturbations
# The first axis will plot the effect of BC perturbations on the
# posterior solution
for i, pert in enumerate(perts): 
    pert_BC = inv.Inversion(
        U=U, gamma=1, BC=true_BC.BCt + pert)
    # print(pert_BC.c)

    # Plot the relative error in the posterior solution
    ax[0].plot(
        pert_BC.xp, 
        (pert_BC.xhat*pert_BC.xa_abs - true_BC.xt_abs)/true_BC.xa_abs/pert,
        color=fp.color(8 - 2*i, lut=12), lw=2, label=f'{pert}')

    # Plot the relative error in the posterior solution
    ax[1].plot(
        pert_BC.xp, (pert_BC.xhat - true_BC.xhat)/true_BC.xa/pert,
        color=fp.color(8 - 2*i, lut=12), lw=2, label=f'{pert}')
    print(f'Real difference : {stats.rel_err(pert_BC.xhat, true_BC.xhat)}')

    # Plot approximations to estimate
    ax[2].plot(pert_BC.xp,
               pert_BC.estimate_delta_xhat(pert)/pert_BC.xa/pert,
               color=fp.color(8 - 2*i, lut=12), lw=2)
    ax[2].plot(pert_BC.xp,
               pert_BC.estimate_delta_xhat_2x2(pert)/pert_BC.xa/pert,
               color=fp.color(8 - 2*i, lut=12), lw=2, ls='--')

    print('-'*30)
    # print(pert_BC.g.sum(axis=1)[0])
    # print(pert_BC.xhat[0])
    ax[3].plot(pert_BC.xp,
               -pert*pert_BC.g.sum(axis=1)/pert_BC.xa/pert,
               color=fp.color(8 - 2*i, lut=12), lw=2)
  
# BC correction
# The third axis will plot the effect of correcting the boundary condition 
# as part of the inversion
# ax2 = ax[2].twinx()
# ax2.set_ylim(-0.05, 0.05)
for i, pert in enumerate(perts):
    pert_opt_BC = inv.Inversion(
        U=U, gamma=1, BC=true_BC.BCt + pert, opt_BC=True)
    print(true_BC.BCt + pert, pert_opt_BC.xhat_BC)
    print(pert_opt_BC.xhat - true_BC.xhat)
    
    ax[4].plot(
        pert_opt_BC.xp, (pert_opt_BC.xhat - true_BC.xhat)/true_BC.xa/pert, 
        color=fp.color(8 - 2*i, lut=12), lw=2, label=f'{pert} ppb')

# Sa_abs scaling
# The second axis will plot the effect of using a buffer grid cell
# ax3 = ax[3].twinx()
# ax3.set_ylim(-0.05, 0.05)
# sfs = np.array([1, 5, 10, 50, 100])
sfs = [100]
for i, sf in enumerate(sfs):
    sa_sf = dc(true_BC.sa)
    sa_sf[0] *= sf**2
    pert_sa_BC = inv.Inversion(
        U=U, sa=sa_sf, BC=true_BC.BCt + pert, gamma=1, opt_BC=False)

    # Plot the relative error in the posterior solution
    ax[5].plot(
        pert_sa_BC.xp, (pert_sa_BC.xhat - true_BC.xhat)/true_BC.xa/pert, 
        color=fp.color(8 - 2*i, lut=12), lw=2, label=f'{sf}')


# # Correction and buffer (Sa_abs scaling)
# # The fourth axis will plot the effect of using a buffer grid cell
# sfs = np.array([1, 5, 10, 50, 100])
# for i, sf in enumerate(sfs):
#     sa_sf = dc(true_BC.sa)
#     sa_sf[0] *= sf**2
#     pert_sa_opt_BC = inv.Inversion(
#         U=U, sa=sa_sf, BC=true_BC.BCt + pert, gamma=1, opt_BC=True)

#     # Plot the relative error in the posterior solution
#     ax[4].plot(
#         pert_sa_opt_BC.xp, 
#         stats.rel_err(pert_sa_opt_BC.xhat, true_BC.xhat),#/pert, 
#         color=fp.color(8 - 2*i, lut=12), lw=2, label=f'{sf}')

fp.add_legend(ax[0], loc='upper left', 
              title='Boundary condition perturbation [ppb]', alignment='left',
              ncol=5, handlelength=1)
fp.add_legend(ax[5], loc='upper left', 
              title='Prior error scaling in buffer grid cell', 
              alignment='left', ncol=5, handlelength=1)

# General formatting
for i in range(6):
    ax[i].set_xticks(np.arange(0, s.nstate+1, 5))
    ax[i].set_xlim(0.5, s.nstate + 0.5)
    for k in range(21):
        ax[i].axvline(k + 0.5, c=fp.color(1), alpha=0.2,
                            ls=':', lw=0.5)
    if i in [1, 3, 5]:
        xlabel = 'State vector element'
    else:
        xlabel = ''

    if i in [0, 1]:
        ylabel = r'$\Delta \hat{x}/x_A$'
    else:
        ylabel = ''

    fp.add_labels(ax[i], xlabel, ylabel)

fp.save_fig(fig, plot_dir, f'constant_BC_{suffix}')