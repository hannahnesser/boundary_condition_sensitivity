import numpy as np
from copy import deepcopy as dc
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import rcParams

# Custom packages
import sys
sys.path.append('.')
import settings as s
from utilities import inversion as inv
from utilities import plot_settings as ps
from utilities import format_plots as fp

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
# U = np.concatenate([np.arange(5, 0, -1), 
#                     np.arange(1, 5, 1)])*24
# U = np.repeat(U, 2)
U = 5*24

if type(U) in [float, int]:
    suffix = 'constwind'
else:
    suffix = 'varwind'

true = inv.Inversion(gamma=1, U=U)
true.estimate_D()
print(true.D)
print(true.D/true.L)
# ## -------------------------------------------------------------------------##
# # Define a standard oscillating perturbation
# ## -------------------------------------------------------------------------##
# def oscillating_bc_pert(t, y, amp, freq, phase):
#     return y + amp*np.sin((2*np.pi/true.t.max())*freq*(t + phase))

# # y-intercept, amplitude, frequency, phase
# BC_pert_std = [true.BC, 10, 2, 0]
# BC_pert_std_v = oscillating_bc_pert(true.t, *BC_pert_std)
# pert_std = inv.Inversion(BC=BC_pert_std_v, gamma=1, U=U)
# pert_std_opt = inv.Inversion(BC=BC_pert_std_v, opt_BC=True, gamma=1, U=U)

# ## -------------------------------------------------------------------------##
# # Test the effect of changed y-intercept, amplitude, frequency, and phase 
# ## -------------------------------------------------------------------------##
# figsize = fp.get_figsize(aspect=2.5, rows=4, cols=6, 
#                          max_width=ps.BASE_WIDTH,
#                          max_height=ps.BASE_HEIGHT)
# fig = plt.figure(figsize=figsize)
# gs = gridspec.GridSpec(4, 6, width_ratios=[1, 0.2, 1, 1, 1, 1])
# ax = []
# for j in [0, 2, 3, 4, 5]:
#     ax_i = [fig.add_subplot(gs[0, j])]
#     for i in range(1, 4):
#         if j <= 2:
#             sharey = ax_i[0]
#         elif j > 2:
#             sharey = ax[1][0]
#         ax_i.append(fig.add_subplot(gs[i, j], sharey=sharey,
#                                           sharex=ax_i[0]))
#     ax.append(ax_i)
# ax = np.array(ax).T

# fp.add_title(ax[0, 0], 'Boundary condition')
# fp.add_title(ax[0, 1], 'Standard inversion')
# fp.add_title(ax[0, 2], 'Boundary condition\ncorrection')
# fp.add_title(ax[0, 3], 'Buffer grid cell')
# fp.add_title(ax[0, 4], 'Buffer grid cell and\nboundary condition\ncorrection')

# ys = np.arange(1900, 1950, 10)
# amps = np.arange(10, 60, 10)
# freqs = np.arange(1, 6)
# phase = np.pi*np.arange(0, 2, 0.4)
# periodics = {'y intercept' : ys, 'Amplitude' : amps, 
#              'Frequency' : freqs, 'Phase' : phase}
# rmse_base = {'mean_pert' : [], 'rmse' : []}
# rmse_base2 = {k : dc(rmse_base) for k, _ in periodics.items()}
# rmse = {k : dc(rmse_base2) for k in 
#         ['Standard', 'Correction', 'Buffer', 'Combination']}

# def tot_err(xhat, truth):
#     return  ((xhat[1:] - truth[1:])**2).sum()**0.5

# for j, (label, vals) in enumerate(periodics.items()):
#     print('-'*70)
#     print(label)
#     # Standard perturbation
#     ax[j, 0].axhline(true.BCt, color='0.6', ls='--', zorder=-1, lw=0.75)

#     # Perturbation
#     # ax[j, 0].fill_between(true.obs_t, 1800, 2000, color='0.6', alpha=0.3)
#     ax[j, 0].set_xlim(true.obs_t.min(), true.t.max())
#     ax[j, 0].set_ylim(1850, 1975)
#     ax[j, 0].set_ylabel('Boundary\ncondition (ppb)')

#     # Labels
#     ax[j, 1].set_ylabel(r'$\Delta\hat{x}$')

#     for i, val in enumerate(vals):
#         BC_pert = dc(BC_pert_std)
#         BC_pert[j] = val
#         BC_pert = oscillating_bc_pert(true.t, *BC_pert)
#         mean_pert = (BC_pert[true.t >= true.obs_t.min()]).mean()
#         ax[j, 0].plot(true.t[true.t > true.obs_t.min()], 
#                       BC_pert[true.t > true.obs_t.min()], 
#                       color=fp.color(8 - 2*i, lut=10), lw=0.75)
#         # for tt in true.obs_t:
#         #     ax[j, 0].axvline(tt, color='grey', lw=0.5, ls='--')
#         # if j >= 2:
#         #     avg = (BC_pert[true.t >= true.obs_t.min()]).mean()
#         #     ax[j, 0].plot([true.obs_t.min(), true.obs_t.max()], [avg, avg],
#         #                   color=fp.color(8 - 2*i, lut=10), lw=0.5, ls='--')

#         # Non optimized BC
#         pert = inv.Inversion(BC=BC_pert, gamma=1, U=U)
#         rmse['Standard'][label]['mean_pert'].append(mean_pert)
#         rmse['Standard'][label]['rmse'].append(tot_err(pert.xhat, true.xhat))
#         ax[j, 1].plot(pert.xp, (pert.xhat - true.xhat)/true.xhat, 
#                       color=fp.color(8 - 2*i, lut=10), ls='-', lw=0.75,
#                       marker='o', ms=2)
#         # delta_signal = ((pert.g - true.g) @ (true.y - true.ya))/true.xhat
#         # delta_noise = -(pert.g @ (pert.c - true.c))/true.xhat
#         # ax[j, 1].plot(pert.xp, delta_signal, color=fp.color(8 - 2*i, lut=10), 
#         #               lw=2, ls='--')
#         # ax[j, 1].plot(pert.xp, delta_noise, color=fp.color(8 - 2*i, lut=10), 
#         #               lw=2, ls=':')
        
#         # Optimized BC
#         pert_opt = inv.Inversion(BC=BC_pert, opt_BC=True, gamma=1, U=U)
#         rmse['Correction'][label]['mean_pert'].append(mean_pert)
#         rmse['Correction'][label]['rmse'].append(tot_err(pert_opt.xhat, true.xhat))
#         ax[j, 2].plot(pert_opt.xp, 
#                       (pert_opt.xhat - true.xhat)/true.xhat, 
#                       color=fp.color(8 - 2*i, lut=10), ls='-', lw=0.75,
#                       marker='o', ms=2)
#         # delta_signal = ((pert_opt.g - true.g) @ (true.y - true.ya))/true.xhat
#         # delta_noise = -(pert_opt.g @ (pert_opt.c - true.c))/true.xhat
#         # ax[j, 2].plot(pert_opt.xp, delta_signal, 
#         #               color=fp.color(8 - 2*i, lut=10), lw=2, ls='--')
#         # ax[j, 2].plot(pert_opt.xp, delta_noise, color=fp.color(8 - 2*i, lut=10), 
#         #               lw=2, ls=':')

#         # Sa first element scaled
#         sa_sf = dc(true.sa)
#         sa_sf[0] *= 100
#         pert_sa_BC = inv.Inversion(sa=sa_sf, BC=BC_pert, gamma=1, U=U)
#         rmse['Buffer'][label]['mean_pert'].append(mean_pert)
#         rmse['Buffer'][label]['rmse'].append(tot_err(pert_sa_BC.xhat, true.xhat))
#         ax[j, 3].plot(pert_sa_BC.xp,
#                       (pert_sa_BC.xhat - true.xhat)/true.xhat, 
#                       color=fp.color(8 - 2*i, lut=10), ls='-', lw=0.75,
#                       marker='o', ms=2)
#         # delta_signal = ((pert_sa_BC.g - true.g) @ (true.y - true.ya))/true.xhat
#         # delta_noise = -(pert_sa_BC.g @ (pert_sa_BC.c - true.c))/true.xhat
#         # ax[j, 3].plot(pert_sa_BC.xp, delta_signal, 
#         #               color=fp.color(8 - 2*i, lut=10), lw=2, ls='--')
#         # ax[j, 3].plot(pert_sa_BC.xp, delta_noise, 
#         #               color=fp.color(8 - 2*i, lut=10), lw=2, ls=':')

#         # SA first element scaled and BC optimized
#         sa_sf = dc(true.sa)
#         sa_sf[0] *= 10
#         pert_sa_BC = inv.Inversion(
#             sa=sa_sf, BC=BC_pert, opt_BC=true, gamma=1, U=U)
#         rmse['Combination'][label]['mean_pert'].append(mean_pert)
#         rmse['Combination'][label]['rmse'].append(tot_err(pert_sa_BC.xhat, true.xhat))
#         ax[j, 4].plot(pert_sa_BC.xp,
#                       (pert_sa_BC.xhat - true.xhat)/true.xhat, 
#                       color=fp.color(8 - 2*i, lut=10), ls='-', lw=0.75,
#                       marker='o', ms=2)
#         # delta_signal = ((pert_sa_BC.g - true.g) @ (true.y - true.ya))/true.xhat
#         # delta_noise = -(pert_sa_BC.g @ (pert_sa_BC.c - true.c))/true.xhat
#         # ax[j, 4].plot(pert_sa_BC.xp, delta_signal,
#         #               color=fp.color(8 - 2*i, lut=10), lw=2, ls='--')
#         # ax[j, 4].plot(pert_sa_BC.xp, delta_noise, 
#         #               color=fp.color(8 - 2*i, lut=10), lw=2, ls=':')

# for i in range(1, 5):
#     # ax[0, i].set_xlim(0, true.nstate)
#     ax[0, i].set_ylim(-0.55, 0.55)
#     ax[3, i].set_xlabel('State vector element')
# ax[-1, 0].set_xlabel('Time (hr)')

# for i in range(4):
#     for j in range(1, 5):
#         ax[i, j].set_xticks(np.arange(0, s.nstate+1, 5))
#         ax[i, j].set_xlim(0.5, s.nstate + 0.5)
#         ax[i, j].axhline(0, c=fp.color(1), alpha=0.2, ls='--', lw=1)
#         ax[i, j].axhline(0.25, c=fp.color(1), alpha=0.2, ls='--', lw=1)
#         ax[i, j].axhline(-0.25, c=fp.color(1), alpha=0.2, ls='--', lw=1)
#         for k in range(3):
#             ax[i, j].axvline((k + 1)*5 + 0.5, c=fp.color(1), alpha=0.2,
#                                 ls=':', lw=0.5)
#         if j != 1:
#             # Turn off y labels
#             plt.setp(ax[i, j].get_yticklabels(), visible=False)
#     if i < 3:
#         for j in range(5):
#             # Turn off x labels
#             plt.setp(ax[i, j].get_xticklabels(), visible=False)

# fp.save_fig(fig, plot_dir, f'oscillating_BC_{suffix}')

# fig, ax = fp.get_figax(aspect=1)
# m = ['^', 'D', 'o', 's']
# for j, method in enumerate(['Standard', 'Correction', 'Buffer', 'Combination']):
#     for i, (label, data) in enumerate(rmse[method].items()):
#         ax.scatter(np.abs(1900 - np.array(data['mean_pert'])), 
#                    data['rmse'],
#                    color='white', 
#                    edgecolor=fp.color(i*2, lut=10), marker=m[j], 
#                    label=f'{label} - {method}')
# fp.add_legend(ax, ncol=4, bbox_to_anchor=[0.5, -0.2], loc='upper center')
# fp.add_labels(ax, 'Mean boundary condition bias (ppm)', 'RMSE (ppm)')
# # ax.set_xlim(0, 10)
# # ax.set_ylim(0, 0.3)
# fp.save_fig(fig, plot_dir, f'oscillating_BC_rmse_{suffix}')

