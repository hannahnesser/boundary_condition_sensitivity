import numpy as np
from copy import deepcopy as dc
import os
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.lines import Line2D

# Custom packages
import sys
sys.path.append('.')
import settings as s
import gcpy as gc
# import forward_model as fm
# import inversion as inv
import inversion_new as inv
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
# Define sensitivity test parameters
## -------------------------------------------------------------------------##
# Default is opt_BC = False
true_BC = inv.Inversion()

x_abs_t_sf = dc(s.x_abs_t)
x_abs_t_sf[0] *= 5
xa_abs_sf = dc(s.xa_abs)
xa_abs_sf[0] *= 5
pert_xa = inv.Inversion(x_abs_t=x_abs_t_sf, xa_abs=xa_abs_sf)

sa_sf = dc(true_BC.sa)
sa_sf[0] *= 5
pert_sa = inv.Inversion(sa=sa_sf)

## -------------------------------------------------------------------------##
# Constant boundary condition perturbations
## -------------------------------------------------------------------------##
# Define the constant perturbations to the boundary condition
perts = [10, 25, 50]
opt_BC = False

# Initialize a summary figure
fig_summ, ax_summ = fp.get_figax()
fig_ils, ax_ils = fp.get_figax(aspect=1)

for i, pert in enumerate(perts):
    pert_BC = inv.Inversion(BC=true_BC.BC_t + pert)

    xa_contrib = pert_BC.a @ pert_BC.xa
    bc_contrib = pert_BC.g @ pert_BC.c

    # Summary plot
    ax_summ.plot(s.xp, np.abs(pert_BC.xhat - true_BC.xhat), 
                 c=fp.color(k=4*i), lw=1, ls='-',
                 label=f'+/-{pert:d} ppb')
    # ax_summ.plot(s.xp, np.abs(-pert*g.sum(axis=1)), 
    #              c=fp.color(k=4*i), lw=2, ls='--')

    ax_ils.plot(bc_contrib/(xa_contrib + bc_contrib), 
                np.abs(pert_BC.xhat - true_BC.xhat)/true_BC.xhat,
                c=fp.color(k=4*i), marker='o', 
                label=f'+/-{pert:d} ppb')

    # Solve inversion with xa_abs_sf
    pert_xa_BC = inv.Inversion(x_abs_t=x_abs_t_sf, xa_abs=xa_abs_sf, 
                                BC=true_BC.BC_t + pert)
    xa_contrib = pert_xa_BC.a @ pert_xa_BC.xa
    bc_contrib = pert_xa_BC.g @ pert_xa_BC.c

    # ax_summ.plot(s.xp, np.abs(pert_xa_BC.xhat - pert_xa.xhat)/pert_xa.xhat, 
    #              c=fp.color(k=4*i), lw=1, ls=':',
    #              label=r'x$_{A, 0}$ scaled')
    # ax_ils.plot(bc_contrib/(xa_contrib + bc_contrib), 
    #             np.abs(pert_xa_BC.xhat - pert_xa.xhat)/pert_xa.xhat,
    #             c=fp.color(k=4*i), marker='o', ls=':',
    #             markeredgecolor=fp.color(k=4*i), markerfacecolor='white',
    #             label=r'x$_{A, 0}$ scaled')

# Add ~10% errors
ax_summ.fill_between([0, s.nstate+1], 0.1, color='grey', alpha=0.2,
                     label=r'$\approx$ 10\% error')

# Add text
fp.add_title(ax_summ, 'Constant Boundary Condition Perturbations')
plot.add_text_label(ax_summ, False)
fp.add_labels(ax_summ, 'State vector element',
              r'$\vert\Delta\hat{x}\vert$ (unitless)')

ax_summ.set_ylim(0, 1)
fig_summ, ax_summ = plot.format_plot(fig_summ, ax_summ, s.nstate)
# ax_summ.set_yscale('log')
# ax_summ.set_ylim(1e-18, 10)
# xs = ax_summ.get_xlim()
# ys = ax_summ.get_ylim()
# ax_summ.set_aspect(0.25, adjustable='box')

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
# ax_summ.set_ylim(0, 0.2)

ax_ils.axhline(0.1, color='grey', ls='--', label=r'$\approx$ 10\% errors')
ax_ils.axvline(0.5, color='grey', ls='--', label=r'50\% BC contribution')

ax_ils.set_yscale('log')
ax_ils.set_ylim(1e-5, 10)
# ax_ils.set_ylim(-0.1, 1)
fp.add_labels(ax_ils,
              'Relative contribution of BC\nto posterior correction',
               r'$\vert\Delta\hat{x}\vert$ (unitless)')
fp.add_legend(ax_ils)

# Save plot
fp.save_fig(fig_summ, plot_dir, f'constant_BC_summary')
fp.save_fig(fig_ils, plot_dir, f'constant_BC_vs_ILS')
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
#         sa = (10**(2*x))*dc(s.sa_vec[opt_BC])
#         xhat, _, _, g = inv.solve_inversion(sa_vec=sa, k=k, ya=ya, 
#                                             opt_BC=opt_BC)
#         sa_effect[i, 0] = inv.band_width(g)
#         sa_effect[i, 2] = inv.e_folding_length(g)

#         # Alter Sa (first n grid cells)
#         sa = dc(s.sa_vec[opt_BC])
#         sa[:(ils_t+1)] *= (10**(2*x))
#         xhat, _, _, g = inv.solve_inversion(sa_vec=sa, k=k, ya=ya, 
#                                             opt_BC=opt_BC)
#         sa_effect[i, 1] = inv.band_width(g)
#         sa_effect[i, 3] = inv.e_folding_length(g)

#         # Alter So
#         so = (10**(2*x))*dc(s.so_vec)
#         xhat, _, _, g = inv.solve_inversion(so_vec=so, k=k, ya=ya, 
#                                             opt_BC=opt_BC)
#         so_effect[i, 0] = inv.band_width(g)
#         so_effect[i, 2] = inv.e_folding_length(g)

#         so = dc(s.so_vec)
#         so[:(ils_t+1)*s.nobs_per_cell] *= (10**(2*x))
#         xhat, _, _, g = inv.solve_inversion(so_vec=so, k=k, ya=ya, 
#                                             opt_BC=opt_BC)
#         # print(np.abs(g.sum(axis=1))*1e3)
#         so_effect[i, 1] = inv.band_width(g)
#         so_effect[i, 3] = inv.e_folding_length(g)

#         # Alter xa
#         xa_abs = dc(s.xa_abs)
#         xa_abs[:(ils_t+1)] *= (10**x)
#         k_xa = dc(k)
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