#%%
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from utilities import inversion as inv
from utilities import format_plots as fp
from utilities import inversion_plot as ip
from utilities import utils, grid

project_dir, config = utils.setup()
data_dir = f'{project_dir}/data/data_OSSE'
plot_dir = f'{project_dir}/plots'

# rcParams['text.usetex'] = True
np.set_printoptions(precision=3, linewidth=300, suppress=True)

# --------------------- ------------------- #
# Plot settings
# ---------------------------------------- #
vir = fp.cmap_trans('viridis', set_bad='0.8')
pla = fp.cmap_trans('plasma', set_bad='0.8')

# Magma_r
mag = fp.cmap_trans('magma_r', set_bad='0.8', reverse=True)
# mag = mpl.colormaps.get_cmap('magma_r')
# mag.set_bad(color='0.8')

# RdBu_r
rdbu = mpl.colormaps.get_cmap('RdBu_r')
rdbu.set_bad(color='0.8')

# Magma_r
# mag = mpl.colormaps.get_cmap('plasma')
# mag.set_bad(colo r='0.8')


# ---------------------------------------- #
# Make a true inversion object
# ---------------------------------------- #
case = 'variable'
pert = np.array([7.5, 10, 10, 12.5])
true_BC = inv.Inversion(gamma=1)
clusters = true_BC.clusters

if case == 'constant':
    true_BC = inv.Inversion(
        xtrue=0.5*np.ones(len(true_BC.xa_abs_full),),
        xa_abs=np.random.RandomState(
            config['random_state']).normal(
                0.5, 0.10, (len(true_BC.xa_abs_full), 1)),
        gamma=1)
    vmin, vmax = 0, 0.75
    cmap = rdbu
    cmap_buffer = 'RdBu_r'
    vmin_diff, vmax_diff = -7, 7
else:
    vmin, vmax = 0, 10
    cmap = mag
    cmap_buffer = fp.cmap_trans('magma_r', reverse=True)
    vmin_diff, vmax_diff = -15, 0

print('Degrees of Freedom in the true case : ', np.trace(true_BC.a))
print('State vector dimension : ', true_BC.nstate)

pert_BC = inv.Inversion(
    xtrue=true_BC.xtrue_full, xa_abs=true_BC.xa_abs_full, BC_pert=pert, 
    gamma=true_BC.gamma)

pert_BC_boundary = inv.Inversion(
    xtrue=true_BC.xtrue_full, xa_abs=true_BC.xa_abs_full, BC_pert=pert, 
    gamma=true_BC.gamma, opt_BC=True, sa_BC=pert.max())

pert_BC_buffer = inv.Inversion(
    xtrue=true_BC.xtrue_full, xa_abs=true_BC.xa_abs_full, BC_pert=pert, 
    gamma=true_BC.gamma, buffer=True, buffer_nclusters=10)

pert_BC_buffer_multi_rows = inv.Inversion(
    xtrue=true_BC.xtrue_full, xa_abs=true_BC.xa_abs_full, BC_pert=pert, 
    gamma=true_BC.gamma, buffer=True, buffer_nclusters=None)

prev = pert_BC.preview(
    sa_bc=pert.max(), plot_dir=plot_dir, plot_str=f'prev_{case}')

# Get five buffer boundary
bounding = pert_BC_buffer.clusters.copy()
bounding = bounding.where(bounding.isin(pert_BC_buffer.cluster_idx + 1), 0)
bounding = bounding.where(bounding == 0, 1)
# #%%

# # ---------------------------------------- #
# # Plot the inversion with the true BC
# # ---------------------------------------- #

# Plot the prior
fig, ax = fp.get_figax(
    cols=4, rows=2, maps=True, lats=clusters.lat, lons=clusters.lon,
    max_width=3.5, max_height=5)
# plt.subplots_adjust(hspace=1)
fig.subplots_adjust(hspace=1, wspace=0.00001)

fig_dist, ax_dist = fp.get_figax(aspect=2)

# Observations
lat_full = np.load(f'{data_dir}/lat_full.npy')
lon_full = np.load(f'{data_dir}/lon_full.npy')
y_g = pd.DataFrame(
    {'lon' : np.round(lon_full/0.3125)*0.3125, 
     'lat' : np.round(lat_full/0.25)*0.25,
     'count' : np.ones(len(lat_full))}
)
y_g = y_g.groupby(['lat', 'lon']).sum()
y_g = xr.Dataset.from_dataframe(y_g).fillna(0)
y_g = y_g.where(true_BC.clusters > 0)

print('Statistics of Observations : ')
print('  Total : ', y_g['count'].sum().values)
print('  Minimum grid cell count : ', y_g['count'].min().values)
print('  Maximum grid cell count : ', y_g['count'].max().values)
print('  Mean grid cell count : ', y_g['count'].mean().values)
print('  Grid cells : ', (y_g['count'] >= 0).sum().values)

            #  lat=slice(clusters_nonzero['lat'].min(), 
            #             clusters_nonzero['lat'].max()),
            #   lon=slice(clusters_nonzero['lon'].min(),
            #             clusters_nonzero['lon'].max()))
c = y_g['count'].plot(ax=ax[0, 2], vmin=0, vmax=250, cmap=pla, 
                      add_colorbar=False, snap=True)
ax[0, 2] = fp.add_title(ax[0, 2], 'Observation density')

cax = fp.add_cax(fig, ax[0, 2], horizontal=True)
cb = fig.colorbar(c, ax=ax[0, 2], cax=cax, orientation='horizontal')
cb = fp.format_cbar(cb, cbar_title='Count',
                    horizontal=True) 

# Emissions
fig, ax[0, 0], c = ip.plot_state(
    true_BC.xtrue, true_BC.clusters,
    title=f'True emissions',
    default_value=np.nan, cmap=vir, vmin=vmin, vmax=vmax, cbar=False, 
    fig_kwargs={'figax' : [fig, ax[0, 0]]})
fig, ax[0, 1], c = ip.plot_state(
    true_BC.xa_abs, true_BC.clusters, 
    title=f'Prior emissions',
    default_value=np.nan, cmap=vir, vmin=vmin, vmax=vmax, cbar=False, 
    fig_kwargs={'figax' : [fig, ax[0, 1]]})

cax = fp.add_cax(fig, ax[0, :2], horizontal=True)
cb = fig.colorbar(c, ax=ax[0, :2], cax=cax, orientation='horizontal')
cb = fp.format_cbar(cb, cbar_title='Methane emissions'r' [kg/km$^2$/hr]',
                    horizontal=True)

# Relative difference
rel_diff = 100*(pert_BC.xhat - true_BC.xhat)/true_BC.xa/pert.mean()
fig, ax[1, 0], c = ip.plot_state(
    rel_diff, true_BC.clusters, 
    title='No correction method',
    default_value=np.nan, cmap=cmap, vmin=vmin_diff, vmax=vmax_diff, cbar=False, 
    fig_kwargs={'figax' : [fig, ax[1, 0]]})
rel_diff = rel_diff.flatten()
xs = np.linspace(-16, 0, 32)
ax_dist.plot(xs, gaussian_kde(rel_diff)(xs),
             color=fp.color(2), label='No correction method')

cax = fp.add_cax(fig, ax[0, 3], horizontal=True)
cb = fig.colorbar(c, ax=ax[0, 3], cax=cax, orientation='horizontal')
cb = fp.format_cbar(cb, 
                    cbar_title='Boundary condition\n'r'induced error [% ppb$^{-1}$]',
                    # cbar_title=r'$\Delta \hat{x}/(x_A \sigma_c)$ [% ppb$^{-1}$]',
                    horizontal=True)
cb.set_ticks(ticks=np.arange(0, -16, -3))

prev = 100*prev/pert.mean()
fig, ax[0, 3], c = ip.plot_state(
    prev, true_BC.clusters,  
    title='Preview',
    default_value=np.nan, cmap=cmap, vmin=vmin_diff, vmax=vmax_diff, cbar=False, 
    fig_kwargs={'figax' : [fig, ax[0, 3]]})

# fig, ax[1, 2], c = ip.plot_state(
#     -100*pert*np.array(pert_BC.g.sum(axis=1))/pert.mean()_BC.xa/pert.mean(), true_BC.clusters, 
#     title='Diagnostic',
#     default_value=np.nan,  cmap=cmap, vmin=vmin_diff, vmax=vmax_diff, cbar=False, 
#     fig_kwargs={'figax' : [fig, ax[1, 2]]})

cax = fp.add_cax(fig, ax[1, :], horizontal=True)
cb = fig.colorbar(c, ax=ax[1, :], cax=cax, orientation='horizontal')
cb = fp.format_cbar(cb, 
                    cbar_title=r'Boundary condition induced error [% ppb$^{-1}$]',
                    # cbar_title=r'$\Delta \hat{x}/(x_A \sigma_c)$ [ppb$^{-1}$]',
                    horizontal=True)
cb.set_ticks(ticks=np.arange(0, -16, -3))

# Correction methods
# Boundary condition method
bc_method = 100*(pert_BC_boundary.xhat[:-4] - true_BC.xhat)
bc_method = bc_method/pert_BC_boundary.xa[:-4]/pert.mean()
fig, ax[1, 1], c = ip.plot_state(
    bc_method,
    true_BC.clusters, 
    title='Boundary method',
    default_value=np.nan, cmap=cmap, vmin=vmin_diff, vmax=vmax_diff, cbar=False, 
    fig_kwargs={'figax' : [fig, ax[1, 1]]})
# ax[1, 1].text(0.05, -0.05, 
#               r'$\hat{x}_N$'f' : {pert_BC_boundary.xhat[-4][0]:.2f} ppb''\n'
#               r'$\hat{x}_S$'f' : {pert_BC_boundary.xhat[-3][0]:.2f} ppb''\n'
#               r'$\hat{x}_E$'f' : {pert_BC_boundary.xhat[-2][0]:.2f} ppb''\n'
#               r'$\hat{x}_W$'f' : {pert_BC_boundary.xhat[-1][0]:.2f} ppb''\n',
#               transform=ax[1, 1].transAxes, 
#               horizontalalignment='left', verticalalignment='bottom')
# cax = fp.add_cax(fig, ax[1, 1],  horizontal=True)
# cb = fig.colorbar(ca, ax=ax[1, 1], cax=cax, orientation='horizontal')
# cb = fp.format_cbar(cb, cbar_title='Averaging kernel\nsensitivities',
#                     horizontal=True) 
rel_diff = bc_method.flatten()
ax_dist.plot(xs, gaussian_kde(rel_diff)(xs),
             color=fp.color(4), label='Boundary method')

# Buffer method
pert_BC_buffer_xhat = grid.clusters_1d_to_2d(
    pert_BC_buffer.xhat, pert_BC_buffer.clusters, default_value=1)
true_BC_xhat = grid.clusters_1d_to_2d(
    true_BC.xhat, true_BC.clusters, default_value=1)
xa = grid.clusters_1d_to_2d(pert_BC_buffer.xa, pert_BC_buffer.clusters,
                            default_value=1)
rel_diff = 100*(pert_BC_buffer_xhat - true_BC_xhat)/xa/pert.mean()
rel_diff = rel_diff.where(
    pert_BC_buffer.clusters.isin(pert_BC_buffer.cluster_idx + 1))
rel_diff.plot(ax=ax[1, 2], 
              cmap=cmap_buffer, vmin=vmin_diff, vmax=vmax_diff, 
              add_colorbar=False)
ax[1, 2] = fp.add_title(ax[1, 2], 'Buffer method\n(clusters)')

rel_diff = rel_diff.where(~rel_diff.isnull(), drop=True).values.flatten()
ax_dist.plot(xs, gaussian_kde(rel_diff)(xs), 
             color=fp.color(6), label='Buffer method\n(clusters)')

# Add clusters
clusters_plot = pert_BC_buffer.clusters.where(
    pert_BC_buffer.clusters.isin(pert_BC_buffer.buffer_idx + 1), 0)
for i, idx in enumerate(np.unique(clusters_plot)):
    clusters_plot = clusters_plot.where(clusters_plot != idx, i + 1)
ax[1, 2] = fp.plot_clusters(clusters_plot, ax=ax[1, 2])

# Buffer method 2
pert_BC_buffer_xhat = grid.clusters_1d_to_2d(
    pert_BC_buffer_multi_rows.xhat, pert_BC_buffer_multi_rows.clusters, 
    default_value=1)
true_BC_xhat = grid.clusters_1d_to_2d(
    true_BC.xhat, true_BC.clusters, default_value=1)
xa = grid.clusters_1d_to_2d(pert_BC_buffer_multi_rows.xa, 
                            pert_BC_buffer_multi_rows.clusters)
rel_diff = 100*(pert_BC_buffer_xhat - true_BC_xhat)/xa/pert.mean()
rel_diff = rel_diff.where(
    pert_BC_buffer_multi_rows.clusters.isin(
        pert_BC_buffer_multi_rows.cluster_idx + 1))
rel_diff.plot(ax=ax[1, 3], 
              cmap=cmap_buffer, vmin=vmin_diff, vmax=vmax_diff, 
              add_colorbar=False)
ax[1, 3] = fp.add_title(ax[1, 3], 'Buffer method\n(no clusters)')

rel_diff = rel_diff.where(~rel_diff.isnull(), drop=True).values.flatten()
ax_dist.plot(xs, gaussian_kde(rel_diff)(xs),
             color=fp.color(8), label='Buffer method\n(no clusters)')

# Add clusters
clusters_plot = pert_BC_buffer_multi_rows.clusters.where(
    pert_BC_buffer_multi_rows.clusters.isin(pert_BC_buffer_multi_rows.buffer_idx + 1), 0)
for i, idx in enumerate(np.unique(clusters_plot)):
    clusters_plot = clusters_plot.where(clusters_plot != idx, i + 1)
ax[1, 3] = fp.plot_clusters(clusters_plot, ax=ax[1, 3])

for i in range(6):
    ax.flatten()[i] = fp.plot_clusters(bounding, ax=ax.flatten()[i],
                                       colors='grey', linestyles='dashed',
                                       linewidths=2)


f = lambda x : '' if x == 0 else f((x - 1) // 26) + chr((x - 1) % 26 + ord('a'))
ys = [0.85]*3 + [0.95]*3
fcs = ['none']*7 + ['white']
for i, axis in enumerate(ax.flatten()):
    axis.text(0.025, 0.975, f'({f(i + 1)})', 
              fontsize=config['label_size']*config['scale'],
              transform=axis.transAxes, ha='left', va='top',
              bbox=dict(boxstyle='square,pad=0.1', ec='none', fc=fcs[i]),
              zorder=100)

# Add contours
# clusters_lim = clusters.where(clusters > 0, drop=True)
# titles = [
#     'True error',
#     'Preview metric',
#     'Diagnostic metric',
# ]

# bound = grid.clusters_1d_to_2d(np.abs(prev), clusters, default_value=0)
# bound_2x2 = grid.clusters_1d_to_2d(np.abs(prev_2x2), clusters, 
#                                    default_value=0)
# # for i, axi in enumerate([ax[1, 0], ax[0, 4], ax[1, 3], ax[1, 4]]):
# for i in range(3):
#     xr.plot.contour(bound, levels=[0, 0.05, 1e3], ax=ax[1, i],
#                     colors=['black'])
#     cs = xr.plot.contourf(bound, levels=[0, 0.05, 1e3], ax=ax[1, i],
#                           colors='none', hatches=[None, '//'],
#                           add_colorbar=False)
    # xr.plot.contour(bound_2x2, levels=[0, 0.05, 1e3], ax=ax[1, i],
    #                 colors=['grey'])
    # cs = xr.plot.contourf(bound_2x2, levels=[0, 0.05, 1e3], ax=ax[1, i],
    #                       colors='none', hatches=[None, '\\'],
    #                       add_colorbar=False)
    # fp.add_title(ax=ax[1, i], title=titles[i])

    
# handles, labels = cs.legend_elements()
# labels=['Relative previewed\nerror >= 0.05']
# fp.add_legend(ax[1, 2], handles=handles[1:], labels=labels, 
#               bbox_to_anchor=(0.5, -0.75), loc='upper center')

# ax[1, 4].remove()
# plt.subplots_adjust(hspace=1)  
# fig.subplots_adjust(hspace=1)  
# fig.subplots_adjust(hspace=1, wspace=0.00001)

fp.save_fig(fig, plot_dir, f'inversion_{case}')


fp.add_legend(ax_dist)
fp.add_labels(ax_dist, r'Boundary condition induced error [% ppb$^{-1}$]', 
              'Count')
fp.save_fig(fig_dist, plot_dir, f'inversion_stats_{case}')

#%%
     