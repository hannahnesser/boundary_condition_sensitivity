import numpy as np
import pandas as pd
import xarray as xr
from utilities import inversion as inv
from utilities import format_plots as fp
from utilities import inversion_plot as ip
from utilities import utils

project_dir, config = utils.setup()
data_dir = f'{project_dir}/data/OSSE'
plot_dir = f'{project_dir}/plots'
clusters = xr.open_dataarray(f'{data_dir}/StateVector.nc')

# ---------------------------------------- #
# Plot settings
# ---------------------------------------- #
vir = fp.cmap_trans('viridis')
pla = fp.cmap_trans('plasma')

# ---------------------------------------- #
# Make a true inversion object
# ---------------------------------------- #
true_BC = inv.Inversion()
print(true_BC.xtrue.mean())
print(true_BC.xa_abs.mean())
true_BC = inv.Inversion(xtrue=3*np.ones(true_BC.nstate,),
                        xa_abs=np.random.RandomState(config['random_state']).normal(3, 0.5, (true_BC.nstate, 1)))

# ---------------------------------------- #
# Plot the true emissions and observations
# ---------------------------------------- #
fig, ax = fp.get_figax(cols=2, maps=True, lats=clusters.lat, lons=clusters.lon)

fig, ax[0], c = ip.plot_state(
    true_BC.xtrue, clusters, title='True emissions',
    default_value=0, cmap=vir, vmin=0, vmax=6, cbar=False, 
    fig_kwargs={'figax' : [fig, ax[0]]})

# Plot the observations
lat_g = np.round(true_BC.lat/0.1)*0.1
lon_g = np.round(true_BC.lon/0.1)*0.1
y_g = pd.DataFrame(
    {'lat' : lat_g.reshape(-1,), 
     'lon' : lon_g.reshape(-1,),
     'y' : true_BC.y.reshape(-1,)})
y_g = xr.Dataset.from_dataframe(y_g.groupby(['lat', 'lon']).mean())
c2 = y_g['y'].plot(ax=ax[1], vmin=1830, vmax=1870, cmap='magma', 
                  add_colorbar=False, snap=True)
ax[1] = fp.add_title(ax[1], 'Pseudo-observations')

cax = fp.add_cax(fig, ax[0], horizontal=True)
cb = fig.colorbar(c, ax=ax[0], cax=cax, orientation='horizontal')
cb = fp.format_cbar(cb, cbar_title='Methane emissions\n'r'(kg/km$^2$/hr)',
                    horizontal=True)

cax = fp.add_cax(fig, ax[1], horizontal=True)
cb = fig.colorbar(c2, ax=ax[1], cax=cax, orientation='horizontal')
cb = fp.format_cbar(cb, cbar_title='Methane column\nconcentration (ppb)',
                    horizontal=True)

fp.save_fig(fig, plot_dir, 'x_true_and_obs')

# ---------------------------------------- #
# Plot the inversion with the true BC
# ---------------------------------------- #
# Plot the prior
fig, ax = fp.get_figax(
    cols=2, rows=2, maps=True, lats=clusters.lat, lons=clusters.lon)
fig.subplots_adjust(hspace=0.01)
fig, ax[0, 0], c = ip.plot_state(
    true_BC.xtrue, clusters, title='True emissions',
    default_value=0, cmap=vir, vmin=0, vmax=6, cbar=False, 
    fig_kwargs={'figax' : [fig, ax[0, 0]]})
fig, ax[0, 1], c = ip.plot_state(
    true_BC.xa_abs, clusters, title='Prior emissions',
    default_value=0, cmap=vir, vmin=0, vmax=6, cbar=False, 
    fig_kwargs={'figax' : [fig, ax[0, 1]]})
fig, ax[1, 0], ca = ip.plot_state(
    np.diagonal(true_BC.a), clusters, title='Information content',
    default_value=0, cmap=pla, vmin=0, vmax=1, cbar=False, 
    fig_kwargs={'figax' : [fig, ax[1, 0]]})
fig, ax[1, 1], c = ip.plot_state(
    true_BC.xhat*true_BC.xa_abs, clusters, title='Posterior emissions',
    default_value=0, cmap=vir, vmin=0, vmax=6, cbar=False, 
    fig_kwargs={'figax' : [fig, ax[1, 1]]})

cax = fp.add_cax(fig, ax[1, 1], horizontal=True)
cb = fig.colorbar(c, ax=ax[1, 1], cax=cax, orientation='horizontal')
cb = fp.format_cbar(cb, cbar_title='Methane emissions\n'r'(kg/km$^2$/hr)',
                    horizontal=True)

cax = fp.add_cax(fig, ax[1, 0], horizontal=True)
cb = fig.colorbar(ca, ax=ax[1, 0], cax=cax, orientation='horizontal')
cb = fp.format_cbar(cb, cbar_title='Averaging kernel\nsensitivities',
                    horizontal=True)

fp.save_fig(fig, plot_dir, 'inversion_true')
