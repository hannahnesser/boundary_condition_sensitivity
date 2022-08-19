import numpy as np
import copy
import os
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.lines import Line2D

# Custom packages
import sys
sys.path.append('.')
import settings as s
import gcpy as gc
import forward_model as fm
import inversion as inv
import plot
import config
config.SCALE = config.PRES_SCALE
config.BASE_WIDTH = config.PRES_WIDTH
config.BASE_HEIGHT = config.PRES_HEIGHT
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
# Update the inversion and model parameters
## -------------------------------------------------------------------------##
optimize_BC = False
print_summary = False

## Modify the observational errror
# s.so_vec[:1*s.nobs_per_cell] = 15*np.sqrt(10)
# s.so_vec[1*s.nobs_per_cell:2*s.nobs_per_cell] = 30
# s.so_vec[2*s.nobs_per_cell:3*s.nobs_per_cell] = 30
# s.so_vec[3*s.nobs_per_cell:4*s.nobs_per_cell] = 30
# s.so_vec[4*s.nobs_per_cell:5*s.nobs_per_cell] = 30

## -------------------------------------------------------------------------##
# Print information about the study
## -------------------------------------------------------------------------##
if print_summary:
    print('-'*20, 'BOUNDARY CONDITION SENSITIVITY STUDY', '-'*20)
    print(f'TIME STEP (hrs)          : {(s.delta_t/3600)}')
    print(f'GRID CELL LIFETIME (hrs) : {(s.tau/3600)}')
    print(f'TRUE EMISSIONS (ppb/hr)  :', s.x_abs_t[0]*3600)
    print(f'STEADY STATE (ppb)       :', s.y0)
    print(f'OBSERVATION TIMES (hrs)  :', s.obs_t/3600)
    print(f'MODEL TIMES (hrs)        :', s.t/3600)
    print('PRIOR EMISSIONS           : ', s.xa_abs*3600)
    print('-'*78)

## -------------------------------------------------------------------------##
# Create initial plots
## -------------------------------------------------------------------------##
# Prior
fig, ax = fp.get_figax(rows=2)
fp.add_title(ax[0], 'Base inversion variables')

# Plot "true " emissions
ax[0].plot(s.xp, s.x_abs_t, c=fp.color(2), ls='--', label='Truth')
ax[0].plot(s.xp, s.xa_abs, #yerr=s.xa_abs*s.sa_vec**0.5,
            c=fp.color(4), marker='.', markersize=10, #capsize=2,
            label=r'Prior($\pm$$\approx$ 50\%)')
ax[0].fill_between(s.xp, s.xa_abs - s.xa_abs*s.sa_vec**0.5,
                   s.xa_abs + s.xa_abs*s.sa_vec**0.5,
                   color=fp.color(2), alpha=0.2, zorder=-1)
handles_0, labels_0 = ax[0].get_legend_handles_labels()
ax[0] = fp.add_labels(ax[0], '', 'Emissions\n(ppb/day)')
ax[0].set_ylim(0, 200)

# Observations
ax[1].plot(s.xp, s.y0, c='black', label='Steady state', zorder=10)
ax[1].plot(s.xp, s.y, c='grey', label='Observations\n($\pm$ 15 ppb)', lw=0.5,
           zorder=9)

# Error range
y_err_min = (s.y - s.so_vec.reshape(s.nstate, s.nobs_per_cell)**0.5).min(axis=1)
y_err_max = (s.y + s.so_vec.reshape(s.nstate, s.nobs_per_cell)**0.5).max(axis=1)
ax[1].fill_between(s.xp, y_err_min, y_err_max, color='grey', alpha=0.2)
handles_1, labels_1 = ax[1].get_legend_handles_labels()
handles_0.extend(handles_1)
labels_0.extend(labels_1)

# Aesthetics
ax[1] = fp.add_legend(ax[1], handles=handles_0, labels=labels_0,
                      bbox_to_anchor=(0.95, 0.5),
                      loc='center left', ncol=1,
                      bbox_transform=fig.transFigure)
ax[1] = fp.add_labels(ax[1], 'State vector element', 'XCH4\n(ppb)')
ax[1].set_ylim(s.y0.min()-50, s.y0.max()+50)
fig, ax = plot.format_plot(fig, ax, s.nstate)
fp.save_fig(fig, plot_dir, f'prior_obs')

## -------------------------------------------------------------------------##
# Solve the inversion with the "true" boundary condition
## -------------------------------------------------------------------------##
# Inversion plots
fig, ax = fp.get_figax(rows=2)
ax[0].plot(s.xp, s.x_abs_t, c=fp.color(2), ls='--', label='Truth')
ax[0].plot(s.xp, s.xa_abs, c=fp.color(4), marker='.', markersize=10, 
           label='Prior')

# Test 1: BC = truth
ls = ['--', '-']
labels = ['BC optimized', 'BC not optimized']
for i, (optimize_BC, K) in enumerate({True : s.K_BC, False : s.K}.items()):
    ya = fm.forward_model(s.xa_abs, s.y0, s.BC_abs_t, s.t, s.U, s.L, s.obs_t)
    # c = ya.flatten() - K @ s.xa
    inv_inputs = [s.xa, s.sa_vec, s.y.flatten(), ya.flatten(), s.so_vec, K]
    xhat_t, _, a_t, g_t = inv.solve_inversion(*inv_inputs, optimize_BC)
    xhat_t_err = np.abs(xhat_t*s.xa_abs/s.x_abs_t - 1)
    # yhat_t = fm.forward_model(xhat_t*s.xa_abs, s.y0, s.BC_abs_t, 
    #                           s.t, s.U, s.L, s.obs_t)
    # bw_base = inv.band_width(g_t*s.xa_abs.reshape(-1, 1))
    # ils_base = inv.influence_length(g_t*s.xa_abs.reshape(-1, 1))
    # base = [bw_base, bw_base, ils_base, ils_base]
    # base = [ils_base, ils_base]

    ax[0].plot(s.xp, xhat_t*s.xa_abs, ls=ls[i], marker='*', markersize=10, 
               c=fp.color(6), label=f'Posterior ({labels[i]})')
    ax[1].plot(s.xp, xhat_t_err, ls=ls[i], marker='*', markersize=10,
               c=fp.color(6))

# Limits
ax[0].set_ylim(0, 200)
ax[1].set_xlim(0, 1)

# Add legend
handles_0, labels_0 = ax[0].get_legend_handles_labels()
ax[1] = fp.add_legend(ax[1], handles=handles_0, labels=labels_0,
                      bbox_to_anchor=(0.5, -0.45), loc='upper center', ncol=2)

# Add labels
ax[0] = fp.add_labels(ax[0], '', 'Emissions\n(ppb/day)')
ax[1] = fp.add_labels(ax[1], 'State vector element', 
                      r'$\vert$ Relative error $\vert$')
r'$\vert\Delta\hat{x}\vert$ (ppb/day)'
ax[0] = fp.add_title(ax[0], 
                     f'True Boundary Condition\n(BC = {s.BC_abs_t:d} ppb)')
fig, ax = plot.format_plot(fig, ax, s.nstate)
fp.save_fig(fig, plot_dir, f'constant_BC_{s.BC_abs_t:d}')

## -------------------------------------------------------------------------##
# Plot gain matrix for a number of So vectors
## -------------------------------------------------------------------------##
# x = np.arange(-1.5, 1.51, 0.01)
# sa_effect = np.zeros((len(x), 4))
# so_effect = np.zeros((len(x), 4))
# inv_inputs = [s.xa, s.sa_vec, s.y.flatten(), ya.flatten(), s.so_vec, s.K]
# xhat_t, _, _, g_t = inv.solve_inversion(*inv_inputs, optimize_BC)
# for ind, i in enumerate(x):
#     # Alter Sa
#     sa = (10**(2*i))*copy.deepcopy(s.sa_vec)
#     inv_inputs[1] = sa
#     xhat, _, _, g = inv.solve_inversion(*inv_inputs, optimize_BC)
#     g = g*s.xa_abs.reshape(-1, 1)
#     sa_effect[ind, 0] = inv.band_width(g)
#     # sa_effect[ind, 2] = inv.influence_length(xhat, xhat_t)

#     sa = copy.deepcopy(s.sa_vec)
#     sa[0] *= (10**(2*i))
#     inv_inputs[1] = sa
#     xhat, _, _, g = inv.solve_inversion(*inv_inputs, optimize_BC)
#     g = g*s.xa_abs.reshape(-1, 1)
#     sa_effect[ind, 1] = inv.band_width(g)
#     # sa_effect[ind, 3] = inv.influence_length(xhat, xhat_t)

#     # Alter So
#     so = (10**(2*i))*copy.deepcopy(s.so_vec)
#     inv_inputs[4] = so
#     xhat, _, _, g = inv.solve_inversion(*inv_inputs, optimize_BC)
#     g = g*s.xa_abs.reshape(-1, 1)
#     so_effect[ind, 0] = inv.band_width(g)
#     # so_effect[ind, 2] = inv.influence_length(xhat, xhat_t)

#     so = copy.deepcopy(s.so_vec)
#     so[:s.nobs_per_cell] *= (10**(2*i))
#     inv_inputs[4] = so
#     xhat, _, _, g = inv.solve_inversion(*inv_inputs, optimize_BC)
#     g = g*s.xa_abs.reshape(-1, 1)
#     so_effect[ind, 1] = inv.band_width(g)
#     # so_effect[ind, 3] = inv.influence_length(xhat, xhat_t)


# print(so_effect)
# print(sa_effect)
# # Plotting
# # ls = ['Lifetime', 'Prior error', 'Observational error']
# # Iterate through band width and influence length scales
# suffix = ['bw', 'ils']
# yaxis = ['Gain matrix band width', 'Influence length scale']
# ylim = [(40, 310), (40, 310), (0.5, 10.5), (0.5, 10.5)]
# fig_summ, ax_summ = fp.get_figax(aspect=2, rows=2, cols=2,
#                                  sharex=True, sharey=True)
# # plt.subplots_adjust(wspace=0.5)
# # We want 0 and 1 to --> 1 and 2 and 3 --> 2
# for i, ax in enumerate(ax_summ.flatten()):
#     if i < 2:
#         # ax.scatter(1, base[i], c='grey', zorder=10, label='Base inversion')
#         ax.plot(10**x, sa_effect[:, i], c=fp.color(8), label='Prior error')
#         ax.plot(10**x, so_effect[:, i], c=fp.color(5), label='Observational error')
#         ax.set_ylim(ylim[i])
#         ax.set_xscale('log')

# # Titles
# ax_summ[0, 0] = fp.add_title(ax_summ[0, 0], 'Full domain scaled')
# ax_summ[0, 1] = fp.add_title(ax_summ[0, 1], 'First grid cell scaled')

# # Axis labels
# ax_summ[0, 0] = fp.add_labels(ax_summ[0, 0], '', 'Gain matrix\nband width')
# ax_summ[0, 1] = fp.add_labels(ax_summ[0, 1], '', '')
# ax_summ[1, 0] = fp.add_labels(ax_summ[1, 0], 'Scale factor',
#                               'Influence\nlength scale')
# ax_summ[1, 1] = fp.add_labels(ax_summ[1, 1], 'Scale factor', '')

# # Legend
# handles_0, labels_0 = ax_summ[0, 0].get_legend_handles_labels()
# ax_summ[0, 0] = fp.add_legend(ax_summ[0, 0], handles=handles_0, labels=labels_0,
#                            bbox_to_anchor=(0.5, -0.1),
#                            loc='upper center', ncol=4,
#                            bbox_transform=fig_summ.transFigure)

# fp.save_fig(fig_summ, plot_dir, f'g_summary')

## -------------------------------------------------------------------------##
# Constant boundary condition perturbations
## -------------------------------------------------------------------------##
optimize_BC = False
perts = [25, 50, 100]
fig_summ, ax_summ = fp.get_figax()
ya_t = fm.forward_model(s.xa_abs, s.y0, s.BC_abs_t, s.t, s.U, s.L, s.obs_t)
inv_inputs_t = [s.xa, s.sa_vec, s.y.flatten(), ya_t.flatten(), s.so_vec, K]
xhat_t, _, a_t, g_t = inv.solve_inversion(*inv_inputs_t, optimize_BC)
print(xhat_t)
for i, pert in enumerate(perts):
    BC = s.BC_abs_t + pert

    # Solve inversion
    ya = fm.forward_model(s.xa_abs, s.y0, BC, s.t, s.U, s.L, s.obs_t)
    inv_inputs = [s.xa, s.sa_vec, s.y.flatten(), ya.flatten(), 
                  s.so_vec, s.K]
    xhat, _, _, g = inv.solve_inversion(*inv_inputs, optimize_BC)
    yhat = fm.forward_model(xhat*s.xa_abs, s.y0, BC, s.t, s.U, s.L, s.obs_t)

    # # Plot inversion
    # fig, ax = plot.plot_inversion(s.xa_abs, xhat, s.x_abs_t, xhat_true=xhat_t,
    #                           optimize_BC=optimize_BC)
    # ax = fp.add_title(ax, f'High Boundary Condition\n(BC = {BC:d} ppb)')
    # fp.save_fig(fig, plot_dir, f'constant_BC_{BC:d}')

    # # Plot observations
    # fig, ax = plot.plot_obs(s.nstate, y, ya, s.y0, s.obs_t, optimize_BC)
    # ax = fp.add_title(ax, f'High Boundary Condition\n(BC = {BC:d} ppb)')
    # fp.save_fig(fig, plot_dir, f'constant_BC_{BC:d}_obs')

    # fig, ax = plot.plot_obs_diff(s.nstate, y, yhat, ya, s.obs_t, optimize_BC)
    # ax = fp.add_title(ax, f'High Boundary Condition\n(BC = {BC:d} ppb)')
    # fp.save_fig(fig, plot_dir, f'constant_BC_{BC:d}_obs_diff')

    # # Plot cost function
    # fig, ax = plot.plot_cost_func(xhat, s.xa_abs, s.sa_vec, yhat, y, s.so_vec,
    #                              s.obs_t, optimize_BC)
    # ax = fp.add_title(ax, f'Cost Function Components\n(BC = {BC:d} ppb)')
    # fp.save_fig(fig, plot_dir, f'constant_BC_{BC:d}_cost_func')

    # Summary plot
    print(xhat)
    ax_summ.plot(s.xp, np.abs(xhat - xhat_t),
                    c=fp.color(k=4*i), lw=1, ls='-',
                    label=f'{pert:d}/-{pert:d} ppb')
    ax_summ.plot(s.xp, np.abs(-pert*g.sum(axis=1)),
                    c=fp.color(k=4*i), lw=2, ls='--')

# Add ~10% errors
ax_summ.fill_between([0, s.nstate+1], 0.1, color='grey', alpha=0.2,
                     label=r'$\approx$ 10\% error')

# Add text
fp.add_title(ax_summ, 'Constant Boundary Condition Perturbations')
plot.add_text_label(ax_summ, optimize_BC)
fp.add_labels(ax_summ, 'State vector element',
              r'$\vert\Delta\hat{x}\vert$ (ppb/day)')

# Legend for summary plot
custom_lines = [Line2D([0], [0], color='grey', lw=1, ls='-'),
                Line2D([0], [0], color='grey', lw=2, ls='--')]
custom_labels = ['Numerical solution', 'Predicted solution']
handles, labels = ax_summ.get_legend_handles_labels()
custom_lines.extend(handles)
custom_labels.extend(labels)
fp.add_legend(ax_summ, handles=custom_lines, labels=custom_labels,
              bbox_to_anchor=(0.5, -0.45), loc='upper center', ncol=3)

# Set limits
ax_summ.set_ylim(0, 1)
fig_summ, ax_summ = plot.format_plot(fig_summ, ax_summ, s.nstate)

# Save plot
fp.save_fig(fig_summ, plot_dir, f'constant_BC_summary')
plt.close()

