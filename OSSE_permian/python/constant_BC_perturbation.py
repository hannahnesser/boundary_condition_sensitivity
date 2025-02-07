#%%
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from utilities import inversion as inv
from utilities import format_plots as fp
from utilities import inversion_plot as ip
from utilities import utils, grid

project_dir, config = utils.setup()
data_dir = f'{project_dir}/data/data_OSSE'
plot_dir = f'{project_dir}/plots'
clusters = xr.open_dataarray(f'{data_dir}/StateVector.nc')
clusters = clusters.where(clusters > 0, drop=True)

# ---------------------------------------- #
# Plot settings
# ---------------------------------------- #
vir = fp.cmap_trans('viridis')
pla = fp.cmap_trans('plasma')

# ---------------------------------------- #
# Make a true inversion object
# ---------------------------------------- #
case = 'variable'
pert = 10
true_BC = inv.Inversion(gamma=1)

if case == 'constant':
    true_BC = inv.Inversion(
        xtrue=0.5*np.ones(true_BC.nstate,),
        xa_abs=np.random.RandomState(
            config['random_state']).normal(0.5, 0.10, (true_BC.nstate, 1)))
    pert_BC = inv.Inversion(
        xtrue=0.5*np.ones(true_BC.nstate,),
        xa_abs=np.random.RandomState(
            config['random_state']).normal(0.5, 0.10, (true_BC.nstate, 1)),
        BC_pert=pert)
    vmin, vmax = 0, 0.5
else:
    pert_BC = inv.Inversion(BC_pert=pert, gamma=1)
    vmin, vmax = 0, 6
    
prev, prev_2x2, g, num, den1, den2, den3, den4, den5 = pert_BC.estimate_delta_xhat(
    clusters, sa_bc=pert, plot_dir=plot_dir, plot_str=f'prev_{case}')

# #%%

# # ---------------------------------------- #
# # Plot the inversion with the true BC
# # ---------------------------------------- #

# Plot the prior
fig, ax = fp.get_figax(
    cols=3, rows=3, maps=True, lats=clusters.lat, lons=clusters.lon,
    max_width=3.5, max_height=5 )
# plt.subplots_adjust(hspace=1)
fig.subplots_adjust(hspace=0.5  , wspace=0.05)

# Observations
lat_full = np.load(f'{data_dir}/lat_full.npy')
lon_full = np.load(f'{data_dir}/lon_full.npy')
y_g = pd.DataFrame(
    {'lon' : np.round(lon_full/0.25)*0.25, 'lat' : np.round(lat_full/0.25)*0.25,
     'count' : np.ones(len(lat_full))}
)
y_g = xr.Dataset.from_dataframe(y_g.groupby(['lat', 'lon']).sum())
c = y_g['count'].plot(ax=ax[0, 2], vmin=0, vmax=160, cmap='magma', 
                      add_colorbar=False, snap=True)
ax[0, 2] = fp.add_title(ax[0, 2], 'Observation density')

cax = fp.add_cax(fig, ax[0, 2], horizontal=True)
cb = fig.colorbar(c, ax=ax[0, 2], cax=cax, orientation='horizontal')
cb = fp.format_cbar(cb, cbar_title='Count',
                    horizontal=True) 

# Emissions
tot = (true_BC.xtrue*true_BC.area/(24*31)*1e-9).sum()
fig, ax[0, 0], c = ip.plot_state(
    true_BC.xtrue, clusters, title=f'True emissions', #({tot:.2f} Tg)',
    default_value=0, cmap=vir, vmin=vmin, vmax=vmax, cbar=False, 
    fig_kwargs={'figax' : [fig, ax[0, 0]]})
tot = (true_BC.xa_abs*true_BC.area/(24*31)*1e-9).sum()
fig, ax[0, 1], c = ip.plot_state(
    0.5*true_BC.xa_abs, clusters, title=f'Prior uncertainty', # ({tot:.2f} Tg)',
    default_value=0, cmap=vir, vmin=vmin, vmax=vmax, cbar=False, 
    fig_kwargs={'figax' : [fig, ax[0, 1]]})
# tot = (true_BC.xhat*true_BC.area/(24*31)*1e-9).sum()
# fig, ax[0, 3], c = ip.plot_state(
#     true_BC.xhat*true_BC.xa_abs, clusters, 
#     title=f'Posterior emissions', # ({tot:.2f} Tg)',
#     default_value=0, cmap='viridis', vmin=vmin, vmax=vmax, cbar=False, 
#     fig_kwargs={'figax' : [fig, ax[0, 3]]})
# tot = (pert_BC.xhat*pert_BC.xa_abs*true_BC.area/(24*31)*1e-9).sum()
# fig, ax[0, 4], c = ip.plot_state(
#     pert_BC.xhat*pert_BC.xa_abs, clusters,   
#     title=f'Posterior emissions with a\n5 ppb BC perturbation', #({tot:.2f} Tg)',
#     default_value=0, cmap='viridis', vmin=vmin, vmax=vmax, cbar=False, 
#     fig_kwargs={'figax' : [fig, ax[0, 4]]})
# fig, ax[1, 3], c2 = ip.plot_state(
#     true_BC.xhat, clusters, 
#     title=f'Posterior emissions', # ({tot:.2f} Tg)',
#     default_value=0, cmap='RdBu_r', vmin=0, vmax=2, cbar=False, 
#     fig_kwargs={'figax' : [fig, ax[1 , 3]]})

# Plot componenets of the estimated preview
# Assume some basic parameters for the estimation of k
U = 4*(60**2/1000) # Wind speed, m/s -> km/hr
Mair = 28.97 # Molar mass dry air, g/mol
MCH4 = 16.01 # Molar mass methane, g/mol # These units cancel out
grav = 9.8/1000*(60**4) # Acceleration due to gravity, m/s2 -> km/hr2
p = 1e5*1000*(60**4) # Surface pressure, Pa = kg/m/s2 -> kg/km/hr2
L = np.sqrt(g['lat_dist']*g['lon_dist']) # km
kL = 1e9*(Mair/MCH4)*L*grav/(U*p)

# fig, ax[2, 0], c2 = ip.plot_state(
#     true_BC.sa.reshape(-1,)/g['so'].values[::-1], clusters, 
#     title=f'Error ratio', # ({tot:.2f} Tg)',
#     default_value=0, cmap=vir, vmin=0, vmax=1, cbar=False, 
#     fig_kwargs={'figax' : [fig, ax[2, 0]]})
# fig, ax[2, 1], c2 = ip.plot_state(
#     (kL * g['xa_abs'])[::-1], clusters, 
#     title=f'kL xa_abs', # ({tot:.2f} Tg)',
#     default_value=0, cmap=vir, vmin=0, vmax=1, cbar=False, 
#     fig_kwargs={'figax' : [fig, ax[2, 1]]})
# fig, ax[2, 2], c2 = ip.plot_state(
#     (kL * g['xa_abs'] * true_BC.sa.reshape(-1,)/g['so'].values)[::-1], clusters, 
#     title=f'Product', # ({tot:.2f} Tg)',
#     default_value=0, cmap='RdBu_r', vmin=-0.1, vmax=0.1, cbar=False, 
#     fig_kwargs={'figax' : [fig, ax[2, 2]]})

# fig, ax[1, -1], c2 = ip.plot_state(
#     num.values, clusters, 
#     title=f'Numerator', # ({tot:.2f} Tg)',
#     default_value=0, cmap=vir, vmin=0, vmax=100, cbar=False, 
#     fig_kwargs={'figax' : [fig, ax[1, -1]]})
# fig, ax[2, 0], c2 = ip.plot_state(
#     den1.values, clusters, 
#     title=f'Denominator 1', # ({tot:.2f} Tg)',
#     default_value=0, cmap=vir, vmin=0, vmax=100, cbar=False, 
#     fig_kwargs={'figax' : [fig, ax[2, 0]]})
# fig, ax[2, 1], c2 = ip.plot_state(
#     den2.values, clusters, 
#     title=f'Denominator 2', # ({tot:.2f} Tg)',
#     default_value=0, cmap=vir, vmin=0, vmax=100, cbar=False, 
#     fig_kwargs={'figax' : [fig, ax[2, 1]]})
# fig, ax[2, 2], c2 = ip.plot_state(
#     den3.values, clusters, 
#     title=f'Denominator 3', # ({tot:.2f} Tg)',
#     default_value=0, cmap=vir, vmin=0, vmax=100, cbar=False, 
#     fig_kwargs={'figax' : [fig, ax[2, 2]]})
# fig, ax[2, 3], c2 = ip.plot_state(
#     den4.values, clusters, 
#     title=f'Denominator 4', # ({tot:.2f} Tg)',
#     default_value=0, cmap=vir, vmin=0, vmax=100, cbar=False, 
#     fig_kwargs={'figax' : [fig, ax[2, 3]]})
# fig, ax[2, 4], c2 = ip.plot_state(
#     den5.values, clusters, 
#     title=f'Denominator 5', # ({tot:.2f} Tg)',
#     default_value=0, cmap=vir, vmin=0, vmax=100, cbar=False, 
#     fig_kwargs={'figax' : [fig, ax[2, 4]]})

# print('Error ratio : ', (true_BC.sa.reshape(-1,)/g['so_i'].values)[:5])
# print('Kx est RT : ', (kL * g['x_i'])[:5].values)

cax = fp.add_cax(fig, ax[0, :2], horizontal=True)
cb = fig.colorbar(c, ax=ax[0, :2   ], cax=cax, orientation='horizontal')
cb = fp.format_cbar(cb, cbar_title='Methane emissions'r' (kg/km$^2$/hr)',
                    horizontal=True)

# # Information content
# fig, ax[1, 1], ca = ip.plot_state(
#     np.diagonal(true_BC.a), clusters, title='Information content',
#     default_value=0, cmap=pla, vmin=0, vmax=1, cbar=False, 
#     fig_kwargs={'figax' : [fig, ax[1, 1]]})


# cax = fp.add_cax(fig, ax[1, 1],  horizontal=True)
# cb = fig.colorbar(ca, ax=ax[1, 1], cax=cax, orientation='horizontal')
# cb = fp.format_cbar(cb, cbar_title='Averaging kernel\nsensitivities',
#                     horizontal=True) 

# Relative difference
fig, ax[1, 0], c = ip.plot_state(
    (pert_BC.xhat - true_BC.xhat)/true_BC.xa/pert, 
    clusters, title='True error',
    default_value=0, cmap='RdBu_r', vmin=-0.1, vmax=0.1, cbar=False, 
    fig_kwargs={'figax' : [fig, ax[1, 0]]})

# prev, prev_2x2, g = pert_BC.estimate_delta_xhat(
#     clusters, sa_bc=pert, plot_dir=plot_dir, plot_str=f'prev_{case}')

scal = 1
fig, ax[1, 1], c = ip.plot_state(
    scal*prev/pert, clusters,  
    title='Preview metric',
    # cmap=fp.cmap_trans('viridis'), vmin=0, vmax=2,
    cmap='RdBu_r', vmin=-0.1, vmax=0.1, 
    default_value=0, cbar=False, fig_kwargs={'figax' : [fig, ax[1, 1]]})

fig, ax[1, 2], c3 = ip.plot_state(
    -pert*np.array(pert_BC.g.sum(axis=1))/pert_BC.xa/pert, 
    clusters, title='Diagnostic metric',
    default_value=0,  cmap='RdBu_r', vmin=-0.1, vmax=0.1, cbar=False, 
    fig_kwargs={'figax' : [fig, ax[1, 2]]})

cax = fp.add_cax(fig, ax[1, :], horizontal=True)
cb = fig.colorbar(c, ax=ax[1, :], cax=cax, orientation='horizontal')
cb = fp.format_cbar(cb, cbar_title=r'$\Delta \hat{x}/x_A$ [unitless]',
                    horizontal=True)


# cax = fp.add_cax(fig, ax[2], horizontal=True)
# cb = fig.colorbar(c3, ax=ax[2], cax=cax, orientation='horizontal')
# cb = fp.format_cbar(cb, cbar_title=r'Diagnosed $\Delta \hat{x}$ [untiless]',
#                     horizontal=True)

# Add contours
# clusters_lim = clusters.where(clusters > 0, drop=True)
titles = [
    'True error',
    'Preview metric',
    'Diagnostic metric',
]

# bound = grid.clusters_1d_to_2d(scal*np.abs(prev), clusters, default_value=0)
# bound_2x2 = grid.clusters_1d_to_2d(scal*np.abs(prev_2x2), clusters, 
#                                    default_value=0)
# # for i, axi in enumerate([ax[0, 3], ax[0, 4], ax[1, 3], ax[1, 4]]):
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
fp.save_fig(fig, plot_dir, f'inversion_{case}')


#%%
     