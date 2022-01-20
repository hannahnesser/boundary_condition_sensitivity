import copy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
import format_plots as fp
import config
config.SCALE = config.PRES_SCALE
config.BASE_WIDTH = config.PRES_WIDTH
config.BASE_HEIGHT = config.PRES_HEIGHT
np.set_printoptions(precision=3, linewidth=300, suppress=True)

plot_dir = '../plots/matrix_structure'


# Define functions
def band_width(k, so, sa):
    g = np.array(sa @ k.T @ np.linalg.inv(k @ sa @ k.T + so))
    g_max = g.max(axis=1).reshape(-1, 1)*1e-3
    band_width = (np.abs(g) > g_max).sum(axis=1).max()
    return band_width

def influence_length(k, so, sa, nobs=100, bc_err=100):
    # calculate influence length scale
    g = np.array(sa @ k.T @ np.linalg.inv(k @ sa @ k.T + so))
    g_sum = g.sum(axis=1).reshape(-1, 1)*nobs*bc_err*3600*24
    ils = (g_sum > 10).sum()
    return(ils)

# Define input parameters

#  XA
nstate = 20
nobs_per_cell = 15 #15
nobs = nstate*nobs_per_cell
xa = 100/(3600*24) # ppb/day in ppb/s
# xa = np.load('../data/x_a.npy') # ppb/s
# print(xa)

# SA
sa_err_base = 50/(3600*24) # absolute error ppb/day
sa_base = np.diag(sa_err_base*np.ones(nstate))**2
# sa_base = np.load('../data/s_a_vec.npy')
# sa_base = np.diag(sa_base)

# SO
so_err_base = 15
so_base = np.diag(so_err_base*np.ones(nobs))**2

# K
U_base = 5/3600 # windspeed (5 km/hr in km/s)
L_base = 25 #12.5 # grid cell length (25 km)
tau_base = L_base/U_base
k_base = tau_base*np.tril(np.ones((nstate, nstate)))
k_base = np.repeat(k_base, nobs_per_cell, axis=0)
# print(k_base)
# k_base = np.load('../data/K_t.npy')
# print(k_base)
# print(k_base[::15, :])

# G
g_base = sa_base @ k_base.T @ np.linalg.inv(k_base @ sa_base @ k_base.T
                                            + so_base)
g_base = np.array(g_base)
g_sum_base = g_base.sum(axis=1).reshape(-1, 1)

# print(g_base == 0)
bw_base = band_width(k_base, so_base, sa_base)
ils_base = influence_length(k_base, so_base, sa_base)
base = [bw_base, bw_base, ils_base, ils_base]

# print(g_base*1e2*1e2*3600) # G * number of obs * perturbation error * unit conversion

x = np.arange(-1, 1.01, 0.01)
tau_effect = np.zeros((len(x), 3))
sa_effect = np.zeros((len(x), 4))
so_effect = np.zeros((len(x), 4))
for ind, i in enumerate(x):
    # Alter lifetime
    k = (10**i)*copy.deepcopy(k_base)
    tau_effect[ind, 0] = band_width(k, so_base, sa_base)
    tau_effect[ind, 2] = influence_length(k, so_base, sa_base)

    # Alter Sa
    sa = (10**(2*i))*copy.deepcopy(sa_base)
    sa_effect[ind, 0] = band_width(k_base, so_base, sa)
    sa_effect[ind, 2] = influence_length(k_base, so_base, sa)

    sa = copy.deepcopy(sa_base)
    sa[0, 0] *= (10**(2*i))
    sa_effect[ind, 1] = band_width(k_base, so_base, sa)
    sa_effect[ind, 3] = influence_length(k_base, so_base, sa)

    # Alter So
    so = (10**(2*i))*copy.deepcopy(so_base)
    so_effect[ind, 0] = band_width(k_base, so, sa_base)
    so_effect[ind, 2] = influence_length(k_base, so, sa_base)

    so = copy.deepcopy(so_base)
    so[0, 0] *= (10**(2*i))
    so_effect[ind, 1] = band_width(k_base, so, sa_base)
    so_effect[ind, 3] = influence_length(k_base, so, sa_base)

# Plotting
suffix = ['bw', 'ils']
yaxis = ['Gain matrix band width', 'Influence length scale']
# ylim = [(0, 10), (0, 10), (0.5, 5.5), (0.5, 5.5)]
fig_summ, ax_summ = fp.get_figax(aspect=2, rows=2, cols=2,
                                 sharex=True, sharey=True)
# plt.subplots_adjust(wspace=0.5)
# We want 0 and 1 to --> 1 and 2 and 3 --> 2
for i, ax in enumerate(ax_summ.flatten()):
    ax.scatter(1, base[i], c='grey', zorder=10, label='Base inversion')
    if i in [0, 2]:
        ax.plot(10**x, tau_effect[:, i], c=fp.color(2), ls='--',
                label='Lifetime', zorder=9)
    ax.plot(10**x, sa_effect[:, i], c=fp.color(8), label='Prior error')
    ax.plot(10**x, so_effect[:, i], c=fp.color(5), label='Observational error')
    # ax.set_ylim(ylim[i])
    ax.set_xscale('log')

# Titles
ax_summ[0, 0] = fp.add_title(ax_summ[0, 0], 'Full domain scaled')
ax_summ[0, 1] = fp.add_title(ax_summ[0, 1], 'First grid cell scaled')

# Axis labels
ax_summ[0, 0] = fp.add_labels(ax_summ[0, 0], '', 'Gain matrix\nband width')
ax_summ[0, 1] = fp.add_labels(ax_summ[0, 1], '', '')
ax_summ[1, 0] = fp.add_labels(ax_summ[1, 0], 'Scale factor',
                              'Influence\nlength scale')
ax_summ[1, 1] = fp.add_labels(ax_summ[1, 1], 'Scale factor', '')

# Axis limits
ax_summ[0, 1].set_ylim(ax_summ[0, 0].get_ylim())
ax_summ[1, 1].set_ylim(ax_summ[1, 0].get_ylim())

# Legend
handles_0, labels_0 = ax_summ[0, 0].get_legend_handles_labels()
ax_summ[0, 0] = fp.add_legend(ax_summ[0, 0], handles=handles_0, labels=labels_0,
                           bbox_to_anchor=(0.5, -0.1),
                           loc='upper center', ncol=2,
                           bbox_transform=fig_summ.transFigure)

fp.save_fig(fig_summ, plot_dir, f'g_summary')
# # Single grid cell adjustments
# fig_summ, ax_summ = fp.get_figax(aspect=1.66, cols=2)
# plt.subplots_adjust(wspace=0.5)
# for i in range(2):
#     ax_summ[i].scatter(1, base[i], c='grey', zorder=20,
#                 label='Base inversion')
#     # ax_summ[i].plot(10**x, tau_effect[:, i], ls='--', c=fp.color(5),
#     #              label='Lifetime')
#     ax_summ[i].plot(10**x, sa_effect[:, i+2], c=fp.color(8),
#                  label='Prior error')
#     ax_summ[i].plot(10**x, so_effect[:, i+2], c=fp.color(5),
#                  label='Observational error')
#     ax_summ[i] = fp.add_labels(ax_summ[i], 'Scale factor', yaxis[i])
#     # ax_summ[i] = fp.add_legend(ax_summ[i], bbox_to_anchor=(0.5, -0.25),
#     #                         loc='upper center', ncol=4)
#     ax_summ[i].set_ylim(ylim[i])
#     ax_summ[i].set_xscale('log')


# handles_0, labels_0 = ax_summ[0].get_legend_handles_labels()
# handles_1, labels_1 = ax_summ[1].get_legend_handles_labels()
# handles_0.extend(handles_1)
# labels_0.extend(labels_1)
# ax_summ[0] = fp.add_legend(ax_summ[0], handles=handles_0, labels=labels_0,
#                            bbox_to_anchor=(0.5, -0.25),
#                            loc='upper center', ncol=4,
#                            bbox_transform=fig_summ.transFigure)

# fp.save_fig(fig_summ, plot_dir, f'g_summary_adj')


# # for U in range(1, 16):
# #     k = (L/(U/3600))*np.tril(np.ones((nstate, nstate)))
# # for so_err in range(0, 16, 1):
#     # so_base = np.diag(so_err*np.ones(nstate))**2
# for i in np.arange(-1, 1.1, 0.1):
#     so_sf = 10**(2*i)
#     # print(so_sf)
#     so_base[0, 0] *= so_sf
# # for sa_err in range(10, 110, 10):
# #     sa_base = np.diag(sa_err/3600*np.ones(nstate))**2
# # for sa_sf in range(1, 10):
# #     sa_base[0, 0] *= sa_sf

#     # Calculate the gain matrix
#     g = np.array(sa_base @ k_base.T @ np.linalg.inv(k_base @ sa_base @ k_base.T
#                                                + so_base))
#     g_sum = g.sum(axis=1).reshape(-1, 1)
#     # print(g.max(), g.min())
#     band_width = (np.abs(g) > g.max(axis=1)*1e-3).sum(axis=1).max()
#     # print(band_width)
#     # print(g_sum.reshape(-1,)[:5])

#     # Plot
#     # Make fig and ax
#     fig = plt.figure()
#     ax_g = plt.subplot2grid((1, nstate+7), (0, 0), colspan=nstate, aspect=1)
#     ax_v = plt.subplot2grid((1, nstate+7), (0, nstate+1), colspan=6)

#     # Plot matrices
#     c = ax_g.imshow(g, cmap='RdBu_r',
#                     norm=SymLogNorm(linthresh=1e-10, vmin=-5e-6, vmax=5e-6,
#                                     base=10))
#     ax_v.plot(g_sum, np.arange(nstate+1, 1, -1), color='black')

#     # Aesthetics and Labels
#     for ax in [ax_g, ax_v]:
#         ax.tick_params(axis='both', which='both', left=False, bottom=False,
#                        labelbottom=False, labelleft=False)

#     ax_g = fp.add_title(ax_g, r'$\mathbf{G}$')
#     ax_g = fp.add_labels(ax_g, 'Observations', 'State Vector')
#     ax_g.text(0.98, 0.95, r'$\mathbf{S}_{\mathrm{O}} = %d$ ppb' % so_base[0, 0],
#     #ax_g.text(0.98, 0.95, r'$\mathbf{S}_{\mathrm{A}} = %d$ ppb/day' % sa_err,
#     # ax_g.text(0.98, 0.95, f'U = {U} km/hr',
#               ha='right', va='top',
#               fontsize=config.LABEL_FONTSIZE*config.SCALE,
#               transform=ax_g.transAxes)

#     ax_v = fp.add_title(ax_v, r'$\sum_j \mathbf{G}_{ij}$')
#     ax_v.set_xlim(0, 5e-5)

#     # # Add colorbar
#     cax = fp.add_cax(fig, ax_g, horizontal=True)
#     cbar = fig.colorbar(c, cax=cax, orientation='horizontal')#, ticks=500*np.arange(0, 5))
#     cbar = fp.format_cbar(cbar, '', horizontal=True)

#     ax_v.set_facecolor(fp.color(5, cmap='RdBu_r', lut=11))
#     # Save plot
#     fp.save_fig(fig, plot_dir, f'g_so_adj_{so_base[0, 0]:02d}')
#                 # facecolor=fp.color(5, cmap='RdBu_r', lut=11))

# # print(g)
# # Need to test both inversions that optimize the boundary condition
# # and inversions that don't
