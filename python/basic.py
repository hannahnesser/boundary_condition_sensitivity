import numpy as np
import copy
import os
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.lines import Line2D

# Custom packages
import sys
sys.path.append('.')
# import settings as s
import gcpy as gc
import forward_model as fm
import inversion as inv
import settings as s
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
# Create initial plots
## -------------------------------------------------------------------------##
# Default is opt_BC = False
true_BC = inv.Inversion(gamma=1)

# Prior
fig, ax = fp.get_figax(rows=2)
fp.add_title(ax[0], 'Base inversion variables')

# Plot "true " emissions
ax[0].plot(true_BC.xp, true_BC.x_abs_t, c=fp.color(2), ls='--', label='Truth')
ax[0].plot(true_BC.xp, true_BC.xa_abs, #yerr=c.xa_abs*c.sa_vec**0.5,
            c=fp.color(4), marker='.', markersize=10, #capsize=2,
            label=r'Prior($\pm$$\approx$ 50\%)')
ax[0].fill_between(true_BC.xp, true_BC.xa_abs - true_BC.xa_abs*true_BC.sa**0.5,
                   true_BC.xa_abs + true_BC.xa_abs*true_BC.sa**0.5,
                   color=fp.color(4), alpha=0.2, zorder=-1)
handles_0, labels_0 = ax[0].get_legend_handles_labels()
ax[0] = fp.add_labels(ax[0], '', 'Emissions\n(ppb/day)')
ax[0].set_ylim(0, 30)

# Observations
ax[1].plot(true_BC.xp, true_BC.y0, c='black', label='Steady state', zorder=10)
ax[1].plot(true_BC.xp, 
           true_BC.y.reshape(true_BC.nstate, true_BC.nobs_per_cell),
           c='grey', label='Observations\n($\pm$ 15 ppb)', lw=0.5, zorder=9)
ax[1].axhline(1900, color=fp.color(4), label='True boundary\ncondition (1900 ppb)')
ax[1].axhline(true_BC.y0.mean(), color=fp.color(6),
              label=f'Mean steady\nstate ({true_BC.y0.mean():.0f} ppb)')

# Error range
y_err_min = (true_BC.y.reshape(true_BC.nstate, true_BC.nobs_per_cell) - 
             true_BC.so.reshape(true_BC.nstate, true_BC.nobs_per_cell)**0.5).min(axis=1)
y_err_max = (true_BC.y.reshape(true_BC.nstate, true_BC.nobs_per_cell) + 
             true_BC.so.reshape(true_BC.nstate, true_BC.nobs_per_cell)**0.5).max(axis=1)
ax[1].fill_between(true_BC.xp, y_err_min, y_err_max, color='grey', alpha=0.2)
handles_1, labels_1 = ax[1].get_legend_handles_labels()
handles_0.extend(handles_1)
labels_0.extend(labels_1)

# Aesthetics
ax[1] = fp.add_legend(ax[1], handles=handles_0, labels=labels_0,
                      bbox_to_anchor=(0.95, 0.5), loc='center left', ncol=1,
                      bbox_transform=fig.transFigure)
ax[1] = fp.add_labels(ax[1], 'State vector element', 'XCH4\n(ppb)')
ax[1].set_ylim(true_BC.y0.min() - 10, true_BC.y0.max() + 10)
fig, ax = plot.format_plot(fig, ax, true_BC.nstate)
fp.save_fig(fig, plot_dir, f'prior_obs')

## -------------------------------------------------------------------------##
# Solve the inversion with the "true_BC" boundary condition
## -------------------------------------------------------------------------##
# Inversion plots
fig, ax = fp.get_figax(rows=2)
ax[0].plot(true_BC.xp, true_BC.x_abs_t, c=fp.color(2), ls='--', label='Truth')
ax[0].plot(true_BC.xp, true_BC.xa_abs, c=fp.color(4), marker='.', markersize=10, 
           label='Prior')

# Test 1: BC = truth
ls = ['--', '-']
labels = ['BC optimized', 'BC not optimized']
for i, optimize_BC in enumerate([true_BC, False]):
    true_BC = inv.Inversion(opt_BC=optimize_BC, gamma=1)
    xhat_t_err = np.abs(true_BC.xhat*true_BC.xa_abs/true_BC.x_abs_t - 1)

    ax[0].plot(true_BC.xp, true_BC.xhat*true_BC.xa_abs, ls=ls[i], 
               marker='*', markersize=10, 
               c=fp.color(6), label=f'Posterior\n({labels[i]})')
    ax[1].plot(true_BC.xp, xhat_t_err, ls=ls[i], marker='*', markersize=10,
               c=fp.color(6))

# Limits
ax[0].set_ylim(0, 30)
ax[1].set_xlim(0, 1)

# Add legend
handles_0, labels_0 = ax[0].get_legend_handles_labels()
ax[1] = fp.add_legend(ax[1], handles=handles_0, labels=labels_0,
                      bbox_to_anchor=(0.95, 0.5), loc='center left', ncol=1,
                      bbox_transform=fig.transFigure)

# Add labels
ax[0] = fp.add_labels(ax[0], '', 'Emissions\n(ppb/day)')
ax[1] = fp.add_labels(ax[1], 'State vector element', 
                      r'$\vert$ Relative error $\vert$')
r'$\vert\Delta\hat{x}\vert$ (ppb/day)'
ax[0] = fp.add_title(ax[0], 
                     f'True Boundary Condition\n(BC = {true_BC.BC_t:d} ppb)')
fig, ax = plot.format_plot(fig, ax, true_BC.nstate)
fp.save_fig(fig, plot_dir, f'constant_BC_{s.BC_t:d}')
