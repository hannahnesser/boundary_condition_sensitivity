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
pert = 10
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

# pert_BC = inv.Inversion(
#     xtrue=true_BC.xtrue_full, xa_abs=true_BC.xa_abs_full, BC_pert=pert, 
#     gamma=true_BC.gamma)

# pert_BC_boundary = inv.Inversion(
#     xtrue=true_BC.xtrue_full, xa_abs=true_BC.xa_abs_full, BC_pert=pert, 
#     gamma=true_BC.gamma, opt_BC=True, sa_BC=10)

pert_BC_buffer = inv.Inversion(
    xtrue=true_BC.xtrue_full, xa_abs=true_BC.xa_abs_full, BC_pert=pert, 
    gamma=true_BC.gamma, buffer=True, buffer_nclusters=10)

# pert_BC_buffer_multi_rows = inv.Inversion(
#     xtrue=true_BC.xtrue_full, xa_abs=true_BC.xa_abs_full, BC_pert=pert, 
#     gamma=true_BC.gamma, buffer=True, buffer_nclusters=None)

# prev = pert_BC.preview(
#     sa_bc=pert, plot_dir=plot_dir, plot_str=f'prev_{case}')

# # Get five buffer boundary
bounding = pert_BC_buffer.clusters.copy()
bounding = bounding.where(bounding.isin(pert_BC_buffer.cluster_idx + 1), 0)
bounding = bounding.where(bounding == 0, 1)
# #%%

# # ---------------------------------------- #
# # Plot the inversion with the true BC
# # ---------------------------------------- #

# Plot the prior
fig, ax = fp.get_figax(
    cols=3, rows=2, maps=True, lats=clusters.lat, lons=clusters.lon,
    max_width=3.5, max_height=5)
fig.subplots_adjust(hspace=1, wspace=0.00001)

# Observations
lat_full = np.load(f'{data_dir}/lat_full.npy')
lon_full = np.load(f'{data_dir}/lon_full.npy')
# y_g = pd.DataFrame(
#     {'lon' : np.round(lon_full/0.3125)*0.3125, 
#      'lat' : np.round(lat_full/0.25)*0.25,
#      'count' : np.ones(len(lat_full))}
# )
# y_g = y_g.groupby(['lat', 'lon']).sum()
# y_g = xr.Dataset.from_dataframe(y_g).fillna(0)
# y_g = y_g.where(true_BC.clusters > 0)

# print('Statistics of Observations : ')
# print('  Total : ', y_g['count'].sum().values)
# print('  Minimum grid cell count : ', y_g['count'].min().values)
# print('  Maximum grid cell count : ', y_g['count'].max().values)
# print('  Mean grid cell count : ', y_g['count'].mean().values)
# print('  Grid cells : ', (y_g['count'] >= 0).sum().values)

#             #  lat=slice(clusters_nonzero['lat'].min(), 
#             #             clusters_nonzero['lat'].max()),
#             #   lon=slice(clusters_nonzero['lon'].min(),
#             #             clusters_nonzero['lon'].max()))
# c = y_g['count'].plot(ax=ax[0, 2], vmin=0, vmax=250, cmap=pla, 
#                       add_colorbar=False, snap=True)
# ax[0, 2] = fp.add_title(ax[0, 2], 'Observation density')

# cax = fp.add_cax(fig, ax[0, 2], horizontal=True)
# cb = fig.colorbar(c, ax=ax[0, 2], cax=cax, orientation='horizontal')
# cb = fp.format_cbar(cb, cbar_title='Count',
#                     horizontal=True) 

# Try also the mean obs - mod difference
diff = true_BC.y - (true_BC.k @ true_BC.xa + true_BC.c)
diff = diff*true_BC.count[:, None]
diff_g = pd.DataFrame(
    {'lon' : true_BC.lon_super,
     'lat' : true_BC.lat_super,
     'diff' : diff.reshape(-1,)}
)
diff_g = diff_g.groupby(['lat', 'lon']).mean()
diff_g = xr.Dataset.from_dataframe(diff_g).fillna(0)
diff_g = diff_g.where(true_BC.clusters > 0)

c = diff_g['diff'].plot(ax=ax[0, 2], vmin=-20, vmax=20, cmap='RdBu_r', 
                         add_colorbar=False, snap=True)
ax[0, 2] = fp.add_title(ax[0, 2], 'Mean (observation - model)\n difference')

cax = fp.add_cax(fig, ax[0, 2], horizontal=True)
cb = fig.colorbar(c, ax=ax[0, 2], cax=cax, orientation='horizontal')
cb = fp.format_cbar(cb, cbar_title='Mean difference [ppb]',
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
cb = fig.colorbar(c, ax=ax[0, :2   ], cax=cax, orientation='horizontal')
cb = fp.format_cbar(cb, cbar_title='Methane emissions'r' [kg/km$^2$/hr]',
                    horizontal=True)

# True posterior
fig, ax[1, 0], c = ip.plot_state(
    true_BC.xhat, true_BC.clusters, 
    title='True posterior\n(relative)',
    default_value=np.nan, cmap='RdBu_r', vmin=0, vmax=2, cbar=False, 
    fig_kwargs={'figax' : [fig, ax[1, 0]]})
# rel_diff = rel_diff.flatten()
# xs = np.linspace(-16, 0, 32)
# ax_dist.plot(xs, gaussian_kde(rel_diff)(xs),
#              color=fp.color(2), label='True error')

cax = fp.add_cax(fig, ax[1, 0], horizontal=True)
cb = fig.colorbar(c, ax=ax[1, 0], cax=cax, orientation='horizontal')
cb = fp.format_cbar(cb, cbar_title='Scale factors',
                    horizontal=True)

# True posterior
fig, ax[1, 1], c = ip.plot_state(
    true_BC.xhat.flatten()*true_BC.xa_abs, true_BC.clusters, 
    title='True posterior\n(absolute)',
    default_value=np.nan, cmap=vir, vmin=vmin, vmax=vmax, cbar=False, 
    fig_kwargs={'figax' : [fig, ax[1, 1]]})

# diff = (true_BC.xhat*true_BC.xa_abs - true_BC.xa_abs)
# print(diff.min(), diff.max())

cax = fp.add_cax(fig, ax[1, 1], horizontal=True)
cb = fig.colorbar(c, ax=ax[1, 1], cax=cax, orientation='horizontal')
cb = fp.format_cbar(cb, cbar_title='Methane emissions'r' [kg/km$^2$/hr]',
                    horizontal=True)

# Averaging kernel
fig, ax[1, 2], c = ip.plot_state(
    np.diagonal(true_BC.a), true_BC.clusters, 
    title='Information content',
    default_value=np.nan, cmap=pla, vmin=0, vmax=1, cbar=False, 
    fig_kwargs={'figax' : [fig, ax[1, 2]]})

# Averaging kernel
cax = fp.add_cax(fig, ax[1, 2],  horizontal=True)
cb = fig.colorbar(c, ax=ax[1, 2], cax=cax, orientation='horizontal')
cb = fp.format_cbar(cb, cbar_title='Averaging kernel\nsensitivities',
                    horizontal=True)

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

fp.save_fig(fig, plot_dir, f'inversion_summary_{case}')

#%%
     