import numpy as np
import pandas as pd
import xarray as xr
from utilities import inversion as inv
from utilities import format_plots as fp
from utilities import inversion_plot as ip
from utilities import utils, grid

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
# Load data
# ---------------------------------------- #
true_BC = inv.Inversion()

k = np.load(f'{data_dir}/{config["k"]}')*1e9 # to ppb

# Identify border grid cells
borders = clusters.where(clusters > 0, drop=True)
borders_lat = borders.sel(lat=[borders['lat'][0], borders['lat'][-1]])
borders_lon = borders.sel(lon=[borders['lon'][0], borders['lon'][-1]])
borders = np.concatenate([borders_lat.values.flatten(),
                          borders_lon.values.flatten()])
borders = (np.unique(borders) - 1).astype(int)

borders2 = clusters.where(clusters > 0, drop=True)
borders2_lat = borders2.isel(lat=[0, 1, -2, -1])
borders2_lon = borders2.isel(lon=[0, 1, -2, -1])
borders2 = np.concatenate([borders2_lat.values.flatten(),
                          borders2_lon.values.flatten()])
borders2 = (np.unique(borders2) - 1).astype(int)

interior2 = np.unique(clusters)[1:]
interior2 = [int(i) - 1 for i in interior2 if i -1 not in borders2]

# Define nstate and nobs
nobs = k.shape[0]
nstate = k.shape[1]

# Process the absolute prior emissions (final units are kg/km2/hr)
x_orig_abs = xr.open_dataset(f'{data_dir}/HEMCO_diagnostics.202005010000.nc')
area = x_orig_abs['AREA']
area = grid.clusters_2d_to_1d(area, clusters).reshape((-1, 1))
x_orig_abs = x_orig_abs['EmisCH4_Total'].squeeze(drop=True)
x_orig_abs = grid.clusters_2d_to_1d(x_orig_abs, clusters)
x_orig_abs = x_orig_abs.reshape((-1, 1))
x_orig_abs *= (1e3)**2*(60*60)

# Get the total 
total_orig_abs = (x_orig_abs*area)[interior2].sum()*(24*31*1e-6*1e-6)
print(f'Original emissions total : {total_orig_abs:.2f} Gg/month')

# Udpate the prior with EDF prior (# Originally kg/m2/s)
x_true_abs = xr.open_dataset(f'{data_dir}/permian_EDF_2019.nc')
x_true_abs = x_true_abs.sel(lat=clusters.where(clusters > 0, drop=True)['lat'],
                  lon=clusters.where(clusters > 0, drop=True)['lon'])
x_true_abs = x_true_abs['EmisCH4_Oil'] + x_true_abs['EmisCH4_Gas']
x_true_abs = grid.clusters_2d_to_1d(x_true_abs, clusters)
x_true_abs = x_true_abs.reshape((-1, 1))
x_true_abs *= (1e3)**2*(60*60)

# Get the total
total_true_abs = (x_true_abs*area)[interior2].sum()*(24*31*1e-6*1e-6)
print((x_true_abs*area).sum()*24*31*1e-6*1e-6)
print((x_true_abs*area)[~borders].sum()*24*31*1e-6*1e-6)
print((x_true_abs*area)[~borders2].sum()*24*31*1e-6*1e-6)

print(f'True emissions total : {total_true_abs:.2f} Gg/month')

# Get statistics on prior emissions
x_orig_abs_mean = x_orig_abs.mean()
x_true_abs_mean = x_true_abs.mean()
x_true_abs_std = x_true_abs.std()
x_true_abs_min = x_true_abs.min()
x_true_abs_max = x_true_abs.max()

print(f'The mean true emissions are {x_true_abs_mean:.2f} kg/km2/hr with a standard deviation of {x_true_abs_std:.2f} kg/km2/hr. (Minimum = {x_true_abs_min:.2f} and maximum = {x_true_abs_max:.2f})')

# Define other entities
xa_abs = 0.5*np.ones((nstate, 1))
xa_abs[x_orig_abs > x_orig_abs_mean] = 3
# xa_abs = dc(x_orig_abs)

# Get the total
total_xa_abs = (xa_abs*area)[interior2].sum()*(24*31*1e-6*1e-6)
print(f'Prior emissions total : {total_xa_abs:.2f} Gg/month')

print(f'The mean prior emissions are {np.mean(xa_abs):.2f} kg/m2/hr with a standard deviation of {np.std(xa_abs):.2f} kg/m2/hr.')

xa = np.ones((nstate, 1))
# xa = np.abs(rs.normal(1.2, 0.5, (nstate, 1)))
sa = 0.5**2*np.ones((nstate, 1))
so = 15**2*np.ones((nobs, 1))

# Adjust Jacobian to be relative in terms of our actual prior
k = k*(xa_abs.T/x_orig_abs.T)

# And get x_true in terms of relative terms
x_true = x_true_abs/xa_abs

# Define the true boundary condition, which is expressed through c
c_true = 1850*np.ones((nobs, 1))

# # Figure out the amount of variance that's appropriate for the 
# # observations
# lat_g = np.round(lat/0.25)*0.25
# lon_g = np.round(lon/0.3125)*0.3125
# y_g = pd.DataFrame({'lat' : lat_g.reshape(-1,),
#                     'lon' : lon_g.reshape(-1,),
#                     'y_true' : y_true.reshape(-1,)})
# y_std = y_g.groupby(['lat', 'lon']).std()
# print('The mean variance at 0.25x0.3125 is : ', y_std.mean())

# Generate pseudo-observations
y = k @ x_true + c_true + np.random.RandomState(config['random_state']).normal(0, 10, (nobs, 1))

def solve_inversion(xa, sa, y, so, k, c):
    # Solve the inversion
    kso_inv = k/so
    sa_inv = np.diag(1/sa.reshape(-1,))

    shat = np.linalg.inv(sa_inv + kso_inv.T @ k)
    g = shat @ kso_inv.T
    a = np.identity(len(xa)) - shat @ sa_inv
    xhat = (xa + g @ (y - k @ xa - c))
    gsum = g.sum(axis=1)

    bc_contrib = (g @ c)
    xa_contrib = (a @ xa)
    tot_correct = bc_contrib + xa_contrib
    zeta = bc_contrib/xhat#/tot_correct

    return xhat, np.diag(a), zeta, gsum

# # Solve the inversion
# # gamma = inv.get_gamma(xa, sa, y, so, k, c_true)
# gamma = 1
# so_g = so/gamma
xhat_true, a, _, _ = solve_inversion(xa, sa, y, so, k, c_true)
# total_xhat_true = (xhat_true*xa_abs*area)[interior2].sum()*(24*31*1e-6*1e-6)
# print(f'True posterior emissions total : {total_xhat_true:.2f} Gg/month')

# print(xhat_true.flatten())
# print(true_BC.xhat.flatten())

# Plot the prior
fig, ax = fp.get_figax(cols=2, rows=2, maps=True, 
                       lats=clusters.lat, lons=clusters.lon)
fig.subplots_adjust(hspace=0.01)
fig, ax[0, 0], c = ip.plot_state(x_true_abs, clusters, title='True emissions',
                               default_value=0, cmap=vir, 
                               vmin=0, vmax=6, cbar=False, 
                               fig_kwargs={'figax' : [fig, ax[0, 0]]})
fig, ax[0, 1], c = ip.plot_state(xa_abs, clusters, title='Prior emissions',
                               default_value=0, cmap=vir, 
                               vmin=0, vmax=6, cbar=False, 
                               fig_kwargs={'figax' : [fig, ax[0, 1]]})
fig, ax[1, 0], ca = ip.plot_state(a, clusters, 
                               title='Information content',
                               default_value=0, cmap=pla, 
                               vmin=0, vmax=1, cbar=False, 
                               fig_kwargs={'figax' : [fig, ax[1, 0]]})
fig, ax[1, 1], c = ip.plot_state(xhat_true*xa_abs, clusters, 
                               title='Posterior emissions',
                               default_value=0, cmap=vir, 
                               vmin=0, vmax=6, cbar=False, 
                               fig_kwargs={'figax' : [fig, ax[1, 1]]})

cax = fp.add_cax(fig, ax[1, 1], horizontal=True)
cb = fig.colorbar(c, ax=ax[1, 1], cax=cax, orientation='horizontal')
cb = fp.format_cbar(cb, cbar_title='Methane emissions\n'r'(kg/km$^2$/hr)',
                    horizontal=True)

cax = fp.add_cax(fig, ax[1, 0], horizontal=True)
cb = fig.colorbar(ca, ax=ax[1, 0], cax=cax, orientation='horizontal')
cb = fp.format_cbar(cb, cbar_title='Averaging kernel\nsensitivities',
                    horizontal=True)

fp.save_fig(fig, '../plots/permian/', 'inversion_true')



# ----------------------------------------------------------------- #
# Boundary condition perturbations
# ----------------------------------------------------------------- #
c = 1840*np.ones((nobs, 1))
# gamma = inv.get_gamma(xa, sa, y, so, k, c)
# gamma = 1
# so_g = so/gamma
xhat, a, zeta, gsum = solve_inversion(xa, sa, y, so, k, c)
# xhat = (xhat*xhat/(xhat - zeta))

# total_xhat_pert = (xhat*xa_abs*area)[interior2].sum()*(24*31*1e-6*1e-6)
# print(f'Perturbed posterior emissions total : {total_xhat_pert:.2f} Gg/month')

# fig, ax = fp.get_figax(cols=3, maps=True, 
#                        lats=clusters.lat, lons=clusters.lon)

# fig, ax[0], c1 = inv.plot_state((xhat - xhat_true)*xa_abs, clusters, 
#                                title=r'$\hat{x} - \hat{x}_T$',
#                                default_value=0, cmap='RdBu_r', 
#                                vmin=-2, vmax=2, cbar=False,
#                                fig_kwargs={'figax' : [fig, ax[0]]})
# ax[0].text(0.05, 0.05, r'$\Sigma \Delta \hat{x}$ = 'f'{(total_xhat_pert - total_xhat_true):.0f} Gg ({100*(total_xhat_pert - total_xhat_true)/total_xhat_true:.2f}\%)', transform=ax[0].transAxes)

# fig, ax[1], c2 = inv.plot_state((xhat - xhat_true)/xhat_true, clusters, 
#                                title=r'$\frac{\hat{x} - \hat{x}_T}{\hat{x}_T}$',
#                                default_value=1, cmap='RdBu_r', 
#                                vmin=0, vmax=2, cbar=False,
#                                fig_kwargs={'figax' : [fig, ax[1]]})
# fig, ax[2], c3 = inv.plot_state(zeta, clusters, 
#                                title=r'$\zeta$',
#                                # title=r'$\Sigma$ G',
#                                default_value=0, cmap=fp.cmap_trans('Reds'), 
#                                cbar=False,
#                                # vmin=0.975, vmax=1, cbar=False,
#                                vmin=0, vmax=150,
#                                fig_kwargs={'figax' : [fig, ax[2]]})

# cax = fp.add_cax(fig, ax[0], horizontal=True)
# cb = fig.colorbar(c1, ax=ax[0], cax=cax, orientation='horizontal')
# cb = fp.format_cbar(cb, cbar_title='Difference\n'r'(kg/km$^2$/hr)',
#                     horizontal=True)

# cax = fp.add_cax(fig, ax[1], horizontal=True)
# cb = fig.colorbar(c2, ax=ax[1], cax=cax, orientation='horizontal')
# cb = fp.format_cbar(cb, cbar_title='Difference\n'r'(unitless)',
#                     horizontal=True)

# cax = fp.add_cax(fig, ax[2], horizontal=True)
# cb = fig.colorbar(c3, ax=ax[2], cax=cax, orientation='horizontal')
# cb = fp.format_cbar(cb, cbar_title=r'$\zeta$ (unitless)',
#                     horizontal=True)
# # cb = fp.format_cbar(cb, cbar_title=r'$\Sigma$ G (ppb$^{-1}$)',
# #                     horizontal=True)# \(ppb$^{-1}$\)')

# fp.save_fig(fig, '../plots/permian/', 'inversion_const_pert')


# # Buffer grid cells
# sa_buffer = dc(sa)
# sa_buffer[borders] = 50**2
# # gamma = inv.get_gamma(xa, sa_buffer, y, so, k, c)
# gamma = 1
# so_g = so/gamma
# xhat, a, zeta, gsum = inv.solve_inversion(xa, sa_buffer, y, so_g, k, c)

# total_xhat_pert = (xhat*xa_abs*area)[interior2].sum()*(24*31*1e-6*1e-6)
# print(f'Perturbed & buffered posterior emissions total : {total_xhat_pert:.2f} Gg/month')


# fig, ax = fp.get_figax(cols=3, maps=True, 
#                        lats=clusters.lat, lons=clusters.lon)

# fig, ax[0], c1 = inv.plot_state((xhat - xhat_true)*xa_abs, clusters, 
#                                title=r'$\hat{x} - \hat{x}_T$',
#                                default_value=0, cmap='RdBu_r', 
#                                vmin=-2, vmax=2, cbar=False,
#                                fig_kwargs={'figax' : [fig, ax[0]]})
# ax[0].text(0.05, 0.05, r'$\Sigma \Delta \hat{x}$ = 'f'{(total_xhat_pert - total_xhat_true):.0f} Gg ({100*(total_xhat_pert - total_xhat_true)/total_xhat_true:.2f}\%)', transform=ax[0].transAxes)

# fig, ax[1], c2 = inv.plot_state((xhat - xhat_true)/xhat_true, clusters, 
#                                title=r'$\frac{\hat{x} - \hat{x}_T}{\hat{x}_T}$',
#                                default_value=0, cmap='RdBu_r', 
#                                vmin=-2, vmax=2, cbar=False,
#                                fig_kwargs={'figax' : [fig, ax[1]]})
# fig, ax[2], c3 = inv.plot_state(zeta, clusters, 
#                                title=r'$\zeta$',
#                                # title=r'$\Sigma$ G',
#                                default_value=0, cmap=fp.cmap_trans('Reds'), 
#                                cbar=False,
#                                # vmin=0.975, vmax=1, cbar=False,
#                                vmin=0, vmax=150,
#                                fig_kwargs={'figax' : [fig, ax[2]]})

# cax = fp.add_cax(fig, ax[0], horizontal=True)
# cb = fig.colorbar(c1, ax=ax[0], cax=cax, orientation='horizontal')
# cb = fp.format_cbar(cb, cbar_title='Difference\n'r'(kg/km$^2$/hr)',
#                     horizontal=True)

# cax = fp.add_cax(fig, ax[1], horizontal=True)
# cb = fig.colorbar(c2, ax=ax[1], cax=cax, orientation='horizontal')
# cb = fp.format_cbar(cb, cbar_title='Difference\n'r'(unitless)',
#                     horizontal=True)

# cax = fp.add_cax(fig, ax[2], horizontal=True)
# cb = fig.colorbar(c3, ax=ax[2], cax=cax, orientation='horizontal')
# cb = fp.format_cbar(cb, cbar_title=r'$\zeta$ (unitless)',
#                     horizontal=True)
# # cb = fp.format_cbar(cb, cbar_title=r'$\Sigma$ G (ppb$^{-1}$)',
# #                     horizontal=True)# \(ppb$^{-1}$\)')

# fp.save_fig(fig, '../plots/permian/', 'inversion_const_pert_buffers')

# # Buffer grid cells
# sa_buffer = dc(sa)
# sa_buffer[borders2] = 50**2
# # gamma = inv.get_gamma(xa, sa_buffer, y, so, k, c)
# gamma = 1
# so_g = so/gamma
# xhat, a, zeta, gsum = inv.solve_inversion(xa, sa_buffer, y, so_g, k, c)

# total_xhat_pert = (xhat*xa_abs*area)[interior2].sum()*(24*31*1e-6*1e-6)
# print(f'Perturbed & buffered (x2) posterior emissions total : {total_xhat_pert:.2f} Gg/month')

# fig, ax = fp.get_figax(cols=3, maps=True, 
#                        lats=clusters.lat, lons=clusters.lon)

# fig, ax[0], c1 = inv.plot_state((xhat - xhat_true)*xa_abs, clusters, 
#                                title=r'$\hat{x} - \hat{x}_T$',
#                                default_value=0, cmap='RdBu_r', 
#                                vmin=-2, vmax=2, cbar=False,
#                                fig_kwargs={'figax' : [fig, ax[0]]})
# ax[0].text(0.05, 0.05, r'$\Sigma \Delta \hat{x}$ = 'f'{(total_xhat_pert - total_xhat_true):.0f} Gg ({100*(total_xhat_pert - total_xhat_true)/total_xhat_true:.2f}\%)', transform=ax[0].transAxes)

# fig, ax[1], c2 = inv.plot_state((xhat - xhat_true)/xhat_true, clusters, 
#                                title=r'$\frac{\hat{x} - \hat{x}_T}{\hat{x}_T}$',
#                                default_value=0, cmap='RdBu_r', 
#                                vmin=-2, vmax=2, cbar=False,
#                                fig_kwargs={'figax' : [fig, ax[1]]})
# fig, ax[2], c3 = inv.plot_state(zeta, clusters, 
#                                title=r'$\zeta$',
#                                # title=r'$\Sigma$ G',
#                                default_value=0, cmap=fp.cmap_trans('Reds'), 
#                                cbar=False,
#                                # vmin=0.975, vmax=1, cbar=False,
#                                vmin=0, vmax=150,
#                                fig_kwargs={'figax' : [fig, ax[2]]})

# cax = fp.add_cax(fig, ax[0], horizontal=True)
# cb = fig.colorbar(c1, ax=ax[0], cax=cax, orientation='horizontal')
# cb = fp.format_cbar(cb, cbar_title='Difference\n'r'(kg/km$^2$/hr)',
#                     horizontal=True)

# cax = fp.add_cax(fig, ax[1], horizontal=True)
# cb = fig.colorbar(c2, ax=ax[1], cax=cax, orientation='horizontal')
# cb = fp.format_cbar(cb, cbar_title='Difference\n'r'(unitless)',
#                     horizontal=True)

# cax = fp.add_cax(fig, ax[2], horizontal=True)
# cb = fig.colorbar(c3, ax=ax[2], cax=cax, orientation='horizontal')
# cb = fp.format_cbar(cb, cbar_title=r'$\zeta$ (unitless)',
#                     horizontal=True)
# # cb = fp.format_cbar(cb, cbar_title=r'$\Sigma$ G (ppb$^{-1}$)',
# #                     horizontal=True)# \(ppb$^{-1}$\)')

# fp.save_fig(fig, '../plots/permian/', 'inversion_const_pert_buffers2')

