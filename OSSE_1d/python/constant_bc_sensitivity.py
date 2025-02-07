import numpy as np
from scipy.linalg import block_diag
import os
from copy import deepcopy as dc
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.lines import Line2D
from collections import OrderedDict

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
# # U = 5*24

# if type(U) in [float, int]:
#     suffix = 'constwind'
# else:
#     suffix = 'varwind'

# true_BC = inv.Inversion(U=U, gamma=1)
# print(f'True Kx : {(true_BC.k @ true_BC.xa)[:5]}')
# kx = true_BC.k @ true_BC.xa
# print(f'True Kx mean : {kx.mean()}')
# # zeta = true_BC.estimate_delta_xhat(sa_bc=10)
# fig, ax = fp.get_figax() 
# ax.plot(cov_xD, label='xD variance', color=fp.color(2, lut=11), ls='-.')
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
fig, ax = fp.get_figax(rows=1, cols=2, aspect=2, max_height=4.5, max_width=7.5,
                       width_ratios=[0.25, 1], sharey=True, sharex=False)
ax = ax.T.flatten()
ax[0].set_ylim(-0.22, 0.025)

# Define the base perturbation
pert = 10

# Create wind speed array
U = np.concatenate([np.arange(5, 0, -1), 
                    np.arange(1, 5, 1)])*24
# U = np.repeat(U, 2)
Us = {'Constant' : 5*24, 'Variable' : U}

for i, (wind_name, U) in enumerate(Us.items()):
    fp.add_title(ax[i], f'{wind_name}''\nwind speed')
    print('-'*70)
    print(wind_name)

    # BC perturbations
    # The first axis will plot the effect of BC perturbations on the
    # posterior solution 
    true_BC = inv.Inversion(U=U, gamma=1)
    pert_BC = inv.Inversion(U=U, gamma=1, BC=true_BC.BCt + pert)

    print(f'True Kx : {(true_BC.k @ true_BC.xa)[:5]}')
    print(true_BC.k/true_BC.xa_abs)
    kx = true_BC.k @ true_BC.xa
    print(f'True Kx mean : {kx.mean()}')

    # Plot the relative error in the posterior solution
    ax[i].plot(
        pert_BC.xp, (pert_BC.xhat - true_BC.xhat)/true_BC.xa/pert,
        color='grey' , lw=5, label=f'True error')

    # Plot approximations to estimate
    ax[i].plot(pert_BC.xp,
                pert_BC.estimate_delta_xhat_2x2(pert)/pert,
                color=fp.color(4, cmap='plasma'), lw=4, ls=':', 
                zorder=20,
                label='Preview metric (2 super-observations)')
    # Plot approximations to estimate
    ax[i].plot(pert_BC.xp,
                pert_BC.estimate_delta_xhat(pert)/pert,
                color=fp.color(1, cmap='plasma'), lw=4, ls=':', 
                zorder=20,
                label='Preview metric (1 super-observation)')
    ax[i].plot(pert_BC.xp,
                -pert*pert_BC.g.sum(axis=1)/pert_BC.xa/pert,
                color=fp.color(7, cmap='plasma'), lw=4, ls=':', 
                zorder=20,
                label='Diagnostic metric')

    # BC correction
    # The third axis will plot the effect of correcting the boundary condition 
    # as part of the inversion
    # This isn't exactly insensitive to the magnitude of the perturbation, but oh
    # well
    # for i, pert in enumerate(perts):
    pert_opt_BC = inv.Inversion(
        U=U, gamma=1, BC=true_BC.BCt + pert, opt_BC=True)
    ax[i].plot(
        pert_opt_BC.xp, (pert_opt_BC.xhat - true_BC.xhat)/true_BC.xa/pert, 
        color=fp.color(4, cmap='viridis'), lw=4, ls='--',
        label=f'Boundary condition correction')

    # Sa_abs scaling
    # The second axis will plot the effect of using a buffer grid cell
    # ax3 = ax[3].twinx()
    # ax3.set_ylim(-0.05, 0.05)
    # sfs = np.array([1, 5, 10, 50, 100])
    # for i, sf in enumerate(sfs):
    sf = 100
    sa_sf = dc(true_BC.sa)
    sa_sf[0] *= sf**2
    pert_sa_BC = inv.Inversion(
        U=U, sa=sa_sf, BC=true_BC.BCt + pert, gamma=1, opt_BC=False)
    ax[i].plot(
        pert_sa_BC.xp, (pert_sa_BC.xhat - true_BC.xhat)/true_BC.xa/pert, 
        color=fp.color(7, cmap='viridis'), lw=4, ls='--',
        label='Buffer cell correction')

    # # Correction and buffer (Sa_abs scaling)
    # # The fourth axis will plot the effect of using a buffer grid cell
    # sfs = np.array([1, 5, 10, 50, 100])
    # for i, sf in enumerate(sfs):
    #     sa_sf = dc(true_BC.sa)
    #     sa_sf[0] *= sf**2
    #     pert_sa_opt_BC = inv.Inversion(
    #         U=U, sa=sa_sf, BC=true_BC.BCt + pert, gamma=1, opt_BC=True)

    #     # Plot the relative error in the posterior solution
    #     ax[2].plot(
    #         pert_sa_opt_BC.xp, 
    #         stats.rel_err(pert_sa_opt_BC.xhat, true_BC.xhat),#/pert, 
    #         color=fp.color(8 - 2*i, lut=12), lw=4, label=f'{sf}')

    # fp.add_legend(ax[i], loc='upper left', 
    #             title='Corrected element', alignment='left',
    #             ncol=1, handlelength=2)
    # fp.add_legend(ax[5], loc='upper left', 
    #               title='Prior error scaling in buffer grid cell', 
    #               alignment='left', ncol=1, handlelength=1)

    # General formatting
    if i == 0:
        ylabel = r'$\Delta \hat{x}/(x_A \sigma_c)$'
        xmax = 5
    else:
        ylabel = ''
        xmax = s.nstate

    print(i, xmax)
    ax[i].set_xticks(np.arange(0, xmax +  1, 5))
    ax[i].axhline(0, c='grey', alpha=0.2, zorder=-10)
    for k in range(xmax + 1):
        ax[i].axvline(k + 0.5, c=fp.color(1, lut=11), alpha=0.2,
                            ls=':', lw=0.5)
    xlabel = 'State vector element'
    fp.add_labels(ax[i], xlabel, ylabel)
    ax[i].set_xlim(0.5, xmax + 0.5)
# ax[0].set_xlim(0.5, 5.5)

# Add legend
handles, labels = ax[0].get_legend_handles_labels()

# Add a blank handle and label 
blank_handle = [Line2D([0], [0], markersize=0, lw=0)]
blank_label = ['']
handles.extend(blank_handle)
labels.extend(blank_label)

# Reorder
reorder = [-1, 0, -1, -1, 
           -1, 1, 2, 3,
           -1, 4, 5, -1]

handles = [handles[i] for i in reorder]
labels = [labels[i] for i in reorder]
labels[4] = 'Metrics : '
labels[8] = 'Correction methods : '

ax[0].legend(handles=handles, labels=labels,
             loc='upper center', alignment='center',  
             bbox_to_anchor=(0.5, -0.2), bbox_transform=fig.transFigure,
             ncol=3, handlelength=2, frameon=False, 
             fontsize=ps.LABEL_FONTSIZE*ps.SCALE)
plt.subplots_adjust(hspace=0.05)
ax[0].set_ylim(-0.22, 0.025)


fp.save_fig(fig, plot_dir, f'constant_BC')