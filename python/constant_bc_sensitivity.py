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
opt_BC = False
print_summary = False

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
# Constant boundary condition perturbations
## -------------------------------------------------------------------------##
# Define the constant perturbations to the boundary condition
perts = [10, 25, 50]

# Initialize a summary figure
fig_summ, ax_summ = fp.get_figax()

# Test out the contribution from different components of the posterior
y_contrib = inv.g_t[opt_BC] @ s.y
xa_contrib = inv.a_t[opt_BC] @ s.xa[opt_BC]
bc_contrib = inv.g_t[opt_BC] @ s.c_t[opt_BC]
tot_adjust = y_contrib - xa_contrib - bc_contrib
print(tot_adjust)
print('y      : ', y_contrib)
print('xa     : ', xa_contrib)
print('bc     : ', bc_contrib)
print('g sum  : ', inv.g_t[opt_BC].sum(axis=1))
# print()
print('-'*50)

# tau_t = inv.e_folding_length(inv.g_t[opt_BC])
# ils_t = np.round(tau_t*3 - 0.5).astype(int)
for i, pert in enumerate(perts):
    BC = s.BC_t + pert

    # Solve inversion
    ya = fm.forward_model(s.xa_abs, s.y0, BC, s.t, s.U, s.L, s.obs_t)
    c = ya - s.K[opt_BC] @ s.xa[opt_BC]
    xhat, _, _, g = inv.solve_inversion(ya=ya, opt_BC=opt_BC)

    y_contrib = g @ s.y
    xa_contrib = g @ s.K[opt_BC] @ s.xa[opt_BC]
    bc_contrib = g @ (ya - s.K[opt_BC] @ s.xa[opt_BC])
    tot_adjust = y_contrib - xa_contrib - bc_contrib
    ils = inv.influence_length(g, s.K[opt_BC], s.xa[opt_BC], c)

    print(tot_adjust)
    print('y      : ', y_contrib)
    print('xa     : ', xa_contrib)
    print('bc     : ', bc_contrib)
    print('g sum  : ', inv.g_t[opt_BC].sum(axis=1))
    print('-'*50)

    fig_bar, ax_bar = fp.get_figax()
    # ax_bar.bar(s.xp - 0.055, y_contrib, width=0.1, color=fp.color(1))
    ax_bar.bar(s.xp, xa_contrib/(xa_contrib + bc_contrib), 
               width=0.3, color=fp.color(1))
    ax_bar.bar(s.xp, bc_contrib/(xa_contrib + bc_contrib), 
               bottom=xa_contrib/(xa_contrib + bc_contrib), 
               width=0.3, color=fp.color(5))
    ax_bar.axvline(ils + 0.5, color=fp.color(8))
    ax_bar.set_ylim(0, 1)
    fig_bar, ax_bar = plot.format_plot(fig_bar, ax_bar, s.nstate)

    # ax_bar.set_yscale('log')
    fp.save_fig(fig_bar, plot_dir, f'G_contributions_{pert}')

    # yhat = fm.forward_model(xhat*s.xa_abs, s.y0, BC, s.t, s.U, s.L, s.obs_t)

    # # Calculate an e-folding lifetime
    # tau = inv.e_folding_length(g)

    # Summary plot
    ax_summ.plot(s.xp, np.abs(xhat - inv.xhat_t[opt_BC]), 
                 c=fp.color(k=4*i), lw=1, ls='-',
                 label=f'{pert:d}/-{pert:d} ppb')
    ax_summ.plot(s.xp, np.abs(-pert*g.sum(axis=1)), 
                 c=fp.color(k=4*i), lw=2, ls='--')
    # ax_summ.plot(s.xp, pert*s.tau**3*(s.sa/s.so)**4*np.arange(s.nstate, 0, -1),
    #              c=fp.color(k=4*i), lw=3, ls=':')
    # ax_summ.plot(s.xp, 
    #              np.abs(xhat - inv.xhat_t[opt_BC])[0]*np.exp(-tau*(s.xp - 1)), 
    #              c=fp.color(k=4*i), lw=1, ls=':')

# Add ~10% errors
ax_summ.fill_between([0, s.nstate+1], 0.1, color='grey', alpha=0.2,
                     label=r'$\approx$ 10\% error')
# ax_summ.fill_between([0, 3*tau_t], 1, color=fp.color(0), alpha=0.1,
#                      label=r'3$\tau$')

# Add text
fp.add_title(ax_summ, 'Constant Boundary Condition Perturbations')
plot.add_text_label(ax_summ, opt_BC)
fp.add_labels(ax_summ, 'State vector element',
              r'$\vert\Delta\hat{x}\vert$ (unitless)')

# Legend for summary plot
custom_lines = [Line2D([0], [0], color='grey', lw=1, ls='-'),
                Line2D([0], [0], color='grey', lw=2, ls='--'),
                Line2D([0], [0], color='grey', lw=1, ls=':')]
custom_labels = ['Numerical solution', 'Predicted solution', 
                 'Exponential decay']
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

## -------------------------------------------------------------------------##
# Plot gain matrix for a number of So vectors
## -------------------------------------------------------------------------##
# # Set up scaling factor range
# xx = np.arange(-1.5, 1.51, 0.01)
# lls = [':', '--', '-']
# yaxis = ['Gain matrix band width', 'Influence length scale']
# ylim = [(0, 1.1), (0, 1.1), (-0.5, 10.5), (-0.5, 10.5)]
# fig_summ, ax_summ = fp.get_figax(aspect=2, rows=2, cols=2,
#                                  sharex='all', sharey='row')

# # Iterate through courant numbers
# for j, c in enumerate([0.1, 0.5, 1]):
#     delta_t = c*s.L/s.U
#     t = np.arange(0, s.init_t + s.total_t + delta_t, delta_t)
#     ya = fm.forward_model(s.xa_abs, s.y0, s.BC_t, t, s.U, s.L, s.obs_t)
#     k = fm.build_jacobian(s.xa_abs, s.y0, s.BC_t, t, s.U, s.L, s.obs_t,
#                           opt_BC=opt_BC)

#     # Initialize arrays
#     sa_effect = np.zeros((len(xx), 4))
#     so_effect = np.zeros((len(xx), 4))
#     xa_effect = np.zeros((len(xx), 4))

#     # Iterate through scaling factors
#     for i, x in enumerate(xx):
#         # Alter Sa (full vector)
#         sa = (10**(2*x))*copy.deepcopy(s.sa_vec[opt_BC])
#         xhat, _, _, g = inv.solve_inversion(sa_vec=sa, k=k, ya=ya, 
#                                             opt_BC=opt_BC)
#         sa_effect[i, 0] = inv.band_width(g)
#         sa_effect[i, 2] = inv.e_folding_length(g)

#         # Alter Sa (first n grid cells)
#         sa = copy.deepcopy(s.sa_vec[opt_BC])
#         sa[:(ils_t+1)] *= (10**(2*x))
#         xhat, _, _, g = inv.solve_inversion(sa_vec=sa, k=k, ya=ya, 
#                                             opt_BC=opt_BC)
#         sa_effect[i, 1] = inv.band_width(g)
#         sa_effect[i, 3] = inv.e_folding_length(g)

#         # Alter So
#         so = (10**(2*x))*copy.deepcopy(s.so_vec)
#         xhat, _, _, g = inv.solve_inversion(so_vec=so, k=k, ya=ya, 
#                                             opt_BC=opt_BC)
#         so_effect[i, 0] = inv.band_width(g)
#         so_effect[i, 2] = inv.e_folding_length(g)

#         so = copy.deepcopy(s.so_vec)
#         so[:(ils_t+1)*s.nobs_per_cell] *= (10**(2*x))
#         xhat, _, _, g = inv.solve_inversion(so_vec=so, k=k, ya=ya, 
#                                             opt_BC=opt_BC)
#         # print(np.abs(g.sum(axis=1))*1e3)
#         so_effect[i, 1] = inv.band_width(g)
#         so_effect[i, 3] = inv.e_folding_length(g)

#         # Alter xa
#         xa_abs = copy.deepcopy(s.xa_abs)
#         xa_abs[:(ils_t+1)] *= (10**x)
#         k_xa = copy.deepcopy(k)
#         k_xa *= xa_abs/s.xa_abs
#         ya_xa = fm.forward_model(xa_abs, s.y0, s.BC_t, t, s.U, s.L, s.obs_t)
#         xhat, _, _, g = inv.solve_inversion(k=k_xa, ya=ya_xa, 
#                                             opt_BC=opt_BC)
#         xa_effect[i, 1] = inv.band_width(g)
#         xa_effect[i, 3] = inv.e_folding_length(g)

#     # Plotting
#     # plt.subplots_adjust(wspace=0.5)
#     # We want 0 and 1 to --> 1 and 2 and 3 --> 2
#     for h, ax in enumerate(ax_summ.flatten()):
#         # ax.scatter(1, base[i], c='grey', zorder=10, label='Base inversion')
#         ax.plot(10**xx, sa_effect[:, h], c=fp.color(8), ls=lls[j], 
#                 label='Prior error')
#         ax.plot(10**xx, so_effect[:, h], c=fp.color(5), ls=lls[j], 
#                 label='Observational error')
#         if h in [1, 3]:
#             ax.plot(10**xx, xa_effect[:, h], c=fp.color(2), ls=lls[j], 
#                     label='Prior')
#         ax.set_ylim(ylim[h])
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