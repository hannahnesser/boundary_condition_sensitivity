#%%
import numpy as np
from copy import deepcopy as dc
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import rcParams
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from collections import OrderedDict

# Custom packages
# import sys
# sys.path.append('.')
# import settings as s
from utilities import inversion as inv
from utilities import plot_settings as ps
from utilities import format_plots as fp
from utilities import stats, utils

# rcParams['text.usetex'] = True
np.set_printoptions(precision=3, linewidth=300, suppress=True)

## -------------------------------------------------------------------------##
# File Locations
## -------------------------------------------------------------------------##
project_dir, config = utils.setup()
data_dir = f'{project_dir}/data/data_OSSE'
plot_dir = f'{project_dir}/plots'
if not os.path.exists(plot_dir):
    os.mkdir(plot_dir)

## -------------------------------------------------------------------------##
# Define the true inversion
## -------------------------------------------------------------------------##
U = np.concatenate([np.arange(7, 3, -1), 
                    np.arange(3, 7, 1)])*24*60*60/1000
# U = np.repeat(U, 2)
# U = 5*24

if type(U) in [float, int]:
    suffix = 'constwind'
else:
    suffix = 'varwind'

true = inv.Inversion(gamma=1, U=U)

## -------------------------------------------------------------------------##
# Define a standard oscillating perturbation
## -------------------------------------------------------------------------##
def oscillating_bc_pert(t, y, amp, freq, phase):
    return y + amp*np.sin((2*np.pi/true.t.max())*freq*(t + phase))

# y-intercept, amplitude, frequency, phase
BC_pert_std = [true.BC.mean(), 5, 2, 0]
BC_pert_std_v = oscillating_bc_pert(true.t, *BC_pert_std)
pert_std = inv.Inversion(BC=BC_pert_std_v, gamma=1, U=U)
pert_std_opt = inv.Inversion(BC=BC_pert_std_v, opt_BC=True, gamma=1, U=U)

## -------------------------------------------------------------------------##
# Test the effect of changed y-intercept, amplitude, frequency, and phase 
## -------------------------------------------------------------------------##
figsize = fp.get_figsize(aspect=2.5, rows=4, cols=7, 
                         max_width=ps.BASE_WIDTH/1.5,
                         max_height=ps.BASE_HEIGHT/1.5)
fig = plt.figure(figsize=figsize)
gs = gridspec.GridSpec(4, 7, width_ratios=[1, 0.2, 1, 1, 1, 1, 1])
ax = []
for j in [0, 2, 3, 4, 5, 6]:
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

fp.add_title(ax[0, 0], 'Boundary condition')
fp.add_title(ax[0, 1], 
             'True error')
fp.add_title(ax[0, 2], 'Preview metric')
fp.add_title(ax[0, 3], 'Diagnostic metric')
fp.add_title(ax[0, 4], 
             'Boundary method')
fp.add_title(ax[0, 5], 
             'Buffer method')

ys = np.arange(1900, 1911, 1)
amps = np.arange(5, 15, 2)
print(amps)
freqs = np.arange(1, 6, 0.25)
phase = np.pi*np.arange(0, 2.1, 0.1)
periodics = {'y-intercept' : ys, 'Amplitude' : amps, 
             'Number of periods' : freqs, 'Phase' : phase}
rmse_base = {'mean_pert' : [], 'rmse' : [], 'rmse_up' : [], 'rmse_down' : []}
rmse_base2 = {k : dc(rmse_base) for k, _ in periodics.items()}
rmse = {k : dc(rmse_base2) for k in 
        ['Standard', 'Boundary', 'Buffer']}

def tot_err(xhat, xa, truth):
    return float(((xhat - truth)**2).mean()**0.5/xa.mean())

means = []
stds = []
firstgrid = []

print(true.xhat)
for j, (label, vals) in enumerate(periodics.items()):
    print('-'*70)
    print(label)
    # Standard perturbation
    ax[j, 0].axhline(true.BCt, color='0.6', ls='--', zorder=-1, lw=2)

    # Perturbation
    # ax[j, 0].fill_between(true.obs_t, 1800, 2000, color='0.6', alpha=0.3)
    ax[j, 0].set_xlim(true.obs_t.min(), true.t.max())
    ax[j, 0].set_ylim(1840, 1970)
    ax[j, 0].set_ylabel('Boundary\ncondition (ppb)')

    # Labels
    ax[j, 1].set_ylabel(r'$\Delta \hat{x}/\hat{x}_A$')

    for i, val in enumerate(vals):
        BC_pert = dc(BC_pert_std)
        BC_pert[j] = val
        BC_pert = oscillating_bc_pert(true.t, *BC_pert)
        mean_pert = float((BC_pert[true.t >= true.obs_t.min()]).mean()) - 1900
        means.append(mean_pert)
        std_pert = ((BC_pert - true.BC)[true.t >= true.obs_t.min()]).std()
        stds.append(std_pert)
        ax[j, 0].plot(true.t[true.t > true.obs_t.min()], 
                      BC_pert[true.t > true.obs_t.min()], 
                      color=fp.color(8 - 2*i, lut=12), lw=2)
        # for tt in true.obs_t:
        #     ax[j, 0].axvline(tt, color='grey', lw=0.5, ls='--')
        # if j >= 2:
        #     avg = (BC_pert[true.t >= true.obs_t.min()]).mean()
        #     ax[j, 0].plot([true.obs_t.min(), true.obs_t.max()], [avg, avg],
        #                   color=fp.color(8 - 2*i, lut=12), lw=0.5, ls='--')

        # Non optimized BC
        pert = inv.Inversion(BC=BC_pert, gamma=1, U=U)
        firstgrid.append(((pert.xhat - true.xhat)/true.xa)[0])
        rmse['Standard'][label]['mean_pert'].append(mean_pert)
        rmse['Standard'][label]['rmse'].append(
            tot_err(pert.xhat, pert.xa, true.xhat))
        rmse['Standard'][label]['rmse_up'].append(
            tot_err(pert.xhat[:7], pert.xa[:7], true.xhat[:7]))
        rmse['Standard'][label]['rmse_down'].append(
            tot_err(pert.xhat[7:], pert.xa[7:], true.xhat[7:]))
        if label == 'y-intercept':
            print('-'*70)
            print(val)
            print(((pert.xhat - true.xhat)**2)[:7])
    
        ax[j, 1].plot(pert.xp, (pert.xhat - true.xhat)/true.xa, 
                      color=fp.color(8 - 2*i, lut=12), lw=2)
        # delta_signal = ((pert.g - true.g) @ (true.y - true.ya))/true.xhat
        # delta_noise = -(pert.g @ (pert.c - true.c))/true.xhat
        # ax[j, 1].plot(pert.xp, delta_signal, color=fp.color(8 - 2*i, lut=12), 
        #               lw=2, ls='--')
        # ax[j, 1].plot(pert.xp, delta_noise, color=fp.color(8 - 2*i, lut=12), 
        #               lw=2, ls=':')
        # ax[j, 1].plot(pert.xp, 10*pert.g.sum(axis=1)/pert.xhat,
        #               color=fp.color(8 - 2*i, lut=12), ls='--', lw=0.5)
        
        # Predicted effect
        # ax[j, 1].plot(pert.xp, 
        #               pert.preview_1d(std_pert + mean_pert), 
        #               color=fp.color(8 - 2*i, lut=12), lw=2)
        ax[j, 2].plot(pert.xp, 
                      pert.preview_2d(std_pert + mean_pert), 
                      color=fp.color(8 - 2*i, lut=12), lw=2)
        ax[j, 3].plot(pert.xp,
                      -(std_pert + mean_pert)*pert.g.sum(axis=1)/pert.xa,
                      color=fp.color(8 - 2*i, lut=12), lw=2)
        
        # Optimized BC
        pert_opt = inv.Inversion(BC=BC_pert, opt_BC=True, gamma=1, U=U)
        rmse['Boundary'][label]['mean_pert'].append(mean_pert)
        rmse['Boundary'][label]['rmse'].append(
            tot_err(pert_opt.xhat, pert_opt.xa, true.xhat))
        rmse['Boundary'][label]['rmse_up'].append(
            tot_err(pert_opt.xhat[:7], pert_opt.xa[:7], true.xhat[:7]))
        rmse['Boundary'][label]['rmse_down'].append(
            tot_err(pert_opt.xhat[7:], pert_opt.xa[7:], true.xhat[7:]))
        ax[j, 4].plot(pert_opt.xp, 
                      (pert_opt.xhat - true.xhat)/true.xa, 
                      color=fp.color(8 - 2*i, lut=12), lw=2)
        # delta_signal = ((pert_opt.g - true.g) @ (true.y - true.ya))/true.xhat
        # delta_noise = -(pert_opt.g @ (pert_opt.c - true.c))/true.xhat
        # ax[j, 2].plot(pert_opt.xp, delta_signal, 
        #               color=fp.color(8 - 2*i, lut=12), lw=2, ls='--')
        # ax[j, 2].plot(pert_opt.xp, delta_noise, color=fp.color(8 - 2*i, lut=12), 
        #               lw=2, ls=':')

        # Sa first element scaled
        sa_sf = dc(true.sa)
        sa_sf[0] *= 100
        pert_sa_BC = inv.Inversion(sa=sa_sf, BC=BC_pert, gamma=1, U=U)
        rmse['Buffer'][label]['mean_pert'].append(mean_pert)
        rmse['Buffer'][label]['rmse'].append(
            tot_err(pert_sa_BC.xhat, pert_sa_BC.xa, true.xhat))
        rmse['Buffer'][label]['rmse_up'].append(
            tot_err(pert.xhat[1:7], pert.xa[1:7], true.xhat[1:7]))
        rmse['Buffer'][label]['rmse_down'].append(
            tot_err(pert.xhat[7:], pert.xa[7:], true.xhat[7:]))
        ax[j, 5].plot(pert_sa_BC.xp,
                      (pert_sa_BC.xhat - true.xhat)/true.xa, 
                      color=fp.color(8 - 2*i, lut=12), lw=2)
        # delta_signal = ((pert_sa_BC.g - true.g) @ (true.y - true.ya))/true.xhat
        # delta_noise = -(pert_sa_BC.g @ (pert_sa_BC.c - true.c))/true.xhat
        # ax[j, 3].plot(pert_sa_BC.xp, delta_signal, 
        #               color=fp.color(8 - 2*i, lut=12), lw=2, ls='--')
        # ax[j, 3].plot(pert_sa_BC.xp, delta_noise, 
        #               color=fp.color(8 - 2*i, lut=12), lw=2, ls=':')

        # # SA first element scaled and BC optimized
        # sa_sf = dc(true.sa)
        # sa_sf[0] *= 10
        # pert_sa_BC = inv.Inversion(
        #     sa=sa_sf, BC=BC_pert, opt_BC=true, gamma=1, U=U)
        # rmse['Combination'][label]['mean_pert'].append(mean_pert)
        # rmse['Combination'][label]['rmse'].append(tot_err(pert_sa_BC.xhat, true.xhat))
        # ax[j, 4].plot(pert_sa_BC.xp,
        #               (pert_sa_BC.xhat - true.xhat)/true.xhat, 
        #               color=fp.color(8 - 2*i, lut=12), lw=2)
        # # delta_signal = ((pert_sa_BC.g - true.g) @ (true.y - true.ya))/true.xhat
        # # delta_noise = -(pert_sa_BC.g @ (pert_sa_BC.c - true.c))/true.xhat
        # # ax[j, 4].plot(pert_sa_BC.xp, delta_signal,
        # #               color=fp.color(8 - 2*i, lut=12), lw=2, ls='--')
        # # ax[j, 4].plot(pert_sa_BC.xp, delta_noise, 
        # #               color=fp.color(8 - 2*i, lut=12), lw=2, ls=':')

for i in range(1, 6):
    # ax[0, i].set_xlim(0, true.nstate)
    ax[0, i].set_ylim(-0.55, 0.55)
    ax[3, i].set_xlabel('State vector element')
ax[-1, 0].set_xlabel('Time (hr)')

for i in range(4):
    for j in range(1, 6):
        ax[i, j].set_xticks(np.arange(0, config['nstate'] + 1, 5))
        ax[i, j].set_xlim(0.5, config['nstate'] + 0.5)
        ax[i, j].axhline(0, c=fp.color(1), alpha=0.2, ls='--', lw=1)
        ax[i, j].axhline(0.25, c=fp.color(1), alpha=0.2, ls='--', lw=1)
        ax[i, j].axhline(-0.25, c=fp.color(1), alpha=0.2, ls='--', lw=1)
        for k in range(3):
            ax[i, j].axvline((k + 1)*5 + 0.5, c=fp.color(1), alpha=0.2,
                                ls=':', lw=1)
        if j != 1:
            # Turn off y labels
            plt.setp(ax[i, j].get_yticklabels(), visible=False)
    if i < 3:
        for j in range(5):
            # Turn off x labels
            plt.setp(ax[i, j].get_xticklabels(), visible=False)

fp.save_fig(fig, plot_dir, f'oscillating_BC_{suffix}')
#%%
fig, ax = fp.get_figax(aspect=1)
m = ['^', 'D', 'o', 's']
for j, method in enumerate(['Standard', 'Boundary', 'Buffer']):
    print(method)
    for i, (label, data) in enumerate(rmse[method].items()):
        mm, b, r, bias = stats.comparison_stats(
            np.abs(np.array(data['mean_pert'])), 
            np.array(data['rmse']))
        ax.scatter(
            np.abs(np.array(data['mean_pert'])), data['rmse'],
            color='white', edgecolor=fp.color(i*2, lut=12), marker=m[j], 
            label=f'{label} - {method} (y = {mm:.2f}x + {b:.2f}, R2 = {r**2:.2f})')
fp.add_legend(ax, ncol=1, bbox_to_anchor=[0.5, -0.2], loc='upper center')
fp.add_labels(ax, 'Mean boundary condition bias (ppm)', 'RMSE (ppm)')
# ax.set_xlim(0, 10)
# ax.set_ylim(-0.1, 0.26)
fp.save_fig(fig, plot_dir, f'oscillating_BC_rmse_{suffix}')


# fig, ax = fp.get_figax(aspect=1)
# ax.scatter(means, firstgrid, color=fp.color(3), marker='o')
# _, _, r, bias = stats.comparison_stats(np.array(means).flatten(), 
#                                        np.array(firstgrid).flatten())
# print(r**2)
# ax.scatter(stds, firstgrid, color=fp.color(5), marker='^')
# plt.show()


# %%
fig = plt.figure(figsize=(13, 18/4))
gs = gridspec.GridSpec(2, 3, height_ratios=[0.33, 1], hspace=0.5, wspace=0.1)

ax = []
for i in range(2):
    ax_i = [fig.add_subplot(gs[i, 0])]
    for j in range(1, 3):
        sharey = ax_i[0]
        if i == 0:
            ax_i.append(fig.add_subplot(gs[i, j], sharey=sharey, sharex=sharey))
        else:
            ax_i.append(fig.add_subplot(gs[i, j], sharey=sharey))
    ax.append(ax_i)
ax = np.array(ax)

# fig, ax = fp.get_figax(rows=2, cols=3, aspect=1.5,
#                        max_width=8, max_height=4,
#                        sharex=False, sharey=True)
# fig.subplots_adjust(wspace=0.25, hspace=0.4)
colors = ['grey',
          fp.color(3, 'viridis'), # boundary
          fp.color(6, 'viridis'), # buffer
          ]
xunits = ['ppb', 'ppb', 'unitless', 'day']

# Plot responses
lw1 = [4, 2, 2]
lw = [4, 3, 3]
alphas = [0.5, 1]
ls = ['-', '--', '--']
# ax = ax.flatten()
for j, method in enumerate(['Standard', 'Boundary', 'Buffer']):
    for i, (label, vals) in enumerate(periodics.items()):
        if label == 'Phase':
            continue
        # ax[i] = fp.add_title(ax[i], label)
        ax[1, i].scatter(
            BC_pert_std[i], 0, marker='v', color='black', s=100,
            label='Baseline perturbation', clip_on=False, zorder=100)
        ax[1, i].plot(
            vals, 
            np.array(rmse[method][label]['rmse_up']),
            color=colors[j], ls=ls[j], alpha=alphas[0], lw=lw1[j])
            # label=f'Upwind RRMSE {method}')
        ax[1, i].plot(
            vals,
            np.array(rmse[method][label]['rmse_down']), 
            color=colors[j], ls=ls[j], lw=lw1[j], alpha=alphas[1],)
            # label=f'Downwind RRMSE {method}')
        if i == 0:
            ylabel = 'RRMSE [unitless]'
        else:
            ylabel = ''
        fp.add_labels(ax[1, i], f'{label} [{xunits[i]}]', ylabel)
        if i != 0:
            ax[1, i].tick_params(labelleft=False, length=0)
ax[1, 0].set_ylim(-0.01, 0.76)
        # if i != 0:
        #     print(i)
        #     ax[1, i].set_yticks([])

# Plot perturbations
ys = np.arange(1900, 1911, 2)
amps = np.arange(5, 17, 2)
freqs = np.arange(1, 6, 1)
periodics = {'y-intercept' : ys, 'Amplitude' : amps, 'Number of periods' : freqs,
             'Phase' : phase}
for j, (label, vals) in enumerate(periodics.items()):
    if label == 'Phase':
        continue
    for i, val in enumerate(vals):
        print(j, label)
        BC_pert = dc(BC_pert_std)
        BC_pert[j] = val
        BC_pert = oscillating_bc_pert(true.t, *BC_pert)
        ax[0, j].plot(true.t[true.t > true.obs_t.min()], 
                      BC_pert[true.t > true.obs_t.min()], 
                      color=fp.color(8 - 2*i, lut=12, cmap='bone'), lw=2)
        if j == 0:
            ylabel = 'Boundary\ncondition\n [ppb]'
        else:
            ylabel = ''
        fp.add_labels(ax[0, j], 'Time [hr]', ylabel)
        ax[0, j].set_xticks(np.arange(10, 22, 5))
        if j != 0:
            ax[0, j].tick_params(labelleft=False, length=0)

ax[0, 0].set_xlim(true.obs_t.min(), true.t.max())
ax[0, 0].set_ylim(1880, 1935)

ax[1, 0].set_xticks(np.arange(1900, 1912, 5))
fig.suptitle('Error induced by periodic boundary condition perturbations')
# Add legend
custom_patches = [Line2D([0], [0], color=colors[0], lw=4, ls='-', alpha=0.75),
                  Line2D([0], [0], color=colors[1], lw=2, ls='--', alpha=0.75),
                  Line2D([0], [0], color=colors[2], lw=2, ls='--', alpha=0.75),
                  Line2D([0], [0], color='grey', lw=3, ls='-', alpha=0.5),
                  Line2D([0], [0], color='grey', lw=3, ls='-', alpha=1)
]
custom_labels = ['True error',
                 'Boundary method',
                 'Buffer method', 
                 'Upwind error',
                 'Downwind error']
                # #'Lu et al. (2022)']
patches, labels = ax[1, 0].get_legend_handles_labels()
custom_patches.extend(patches)
custom_labels.extend(labels)
custom_labels = OrderedDict(zip(custom_labels, custom_patches))
custom_patches = custom_labels.values()
custom_labels = custom_labels.keys()
ax[0, 0].legend(handles=custom_patches, labels=custom_labels, frameon=False,
          bbox_to_anchor=(0.5, -0.1), loc='upper center', ncol=2,
          bbox_transform=fig.transFigure)

f = lambda x : '' if x == 0 else f((x - 1) // 26) + chr((x - 1) % 26 + ord('a'))
ys = [0.85]*3 + [0.95]*3
for i, axis in enumerate(ax.flatten()):
    axis.text(0.025, ys[i], f'({f(i + 1)})', 
              fontsize=ps.LABEL_FONTSIZE*ps.SCALE,
             transform=axis.transAxes, ha='left', va='top')

fp.save_fig(fig, plot_dir, f'oscillating_BC_rmse_{suffix}')
# %%
