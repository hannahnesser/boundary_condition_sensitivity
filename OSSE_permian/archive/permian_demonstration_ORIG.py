local = True

if not local:
    data_dir = '/n/jacob_lab/Lab/seasasfs02/dvaron/GEOSChem/files_for_hannah'
    code_dir = '/n/home04/hnesser/boundary_condition_sensitivity/python'
    output_dir = '/n/home04/hnesser/boundary_condition_sensitivity/data'

    import glob
    import numpy as np
    import sys
    sys.path.append(code_dir)
    import gcpy as gc

    # Create a list of Jacobian files
    files = glob.glob(f'{data_dir}/week1/*.pkl')
    files.sort()

    # Load the cluster file
    cluster = f'{data_dir}/Clusters_permian_kmeans.nc'
    cluster = gc.read_file(cluster)

    # Get latitude and longitdue limits
    lats = cluster.lat.values
    lat_delta = np.diff(lats)[0]
    lons = cluster.lon.values
    lon_delta = np.diff(lons)[0]

    # Remove buffer grid cells
    lats, lons = gc.adjust_grid_bounds(lats[0], lats[-1], lat_delta,
                                       lons[0], lons[-1], lon_delta,
                                       buffer=[3, 3, 3, 3])

    # Initialize Jacobian
    K = []
    y_diff = []
    ya = []
    y = []

    # Iterate through the Jacobian files
    for f in files:
        # Load the data and check that there are observations
        data = gc.read_file(f)
        if data['obs_GC'].shape[0] == 0:
            continue

        # Get index to filter out buffer cells
        obs = data['obs_GC']
        filt = np.where((obs[:, 3] >= lats[0]) & (obs[:, 3] <= lats[1]) &
                        (obs[:, 2] >= lons[0]) & (obs[:, 2] <= lons[1]))[0]
        if len(filt) == 0:
            continue

        # Load Jacobian (ppb) and append it to K
        K_f = data['KK'][filt, :]*1e9
        K.append(K_f)

        # Load observations
        y.append(obs[:, 0][filt])
        ya.append(obs[:, 1][filt])

        # Load the observation-model difference (ppb) and append it to y_diff
        y_diff.append((obs[:, 0] - obs[:, 1])[filt])

    # Concatenate to form the full Jacobian
    K = np.concatenate(K)
    y_diff = np.concatenate(y_diff)
    ya = np.concatenate(ya)
    y = np.concatenate(y)

    # Save out
    np.save(f'{output_dir}/K_permian.npy', K)
    np.save(f'{output_dir}/y_diff_permian.npy', y_diff)
    np.save(f'{output_dir}/ya_permian.npy', ya)
    np.save(f'{output_dir}/y_permian.npy', y)

else:
    data_dir = '../data'
    plot_dir = '../plots/permian/'
    code_dir = '.'
    # output_dir = '/n/home04/hnesser/boundary_condition_sensitivity/data'

    import glob
    import numpy as np
    import xarray as xr
    import matplotlib.pyplot as plt
    import matplotlib.colors as colors
    import copy
    import sys
    sys.path.append(code_dir)
    import gcpy as gc
    import inversion as inv
    import plot
    import format_plots as fp
    import plot_settings as ps
    small_map_kwargs = {'draw_labels' : False}
    ps.SCALE = ps.PRES_SCALE
    ps.BASE_WIDTH = ps.PRES_WIDTH
    ps.BASE_HEIGHT = ps.PRES_HEIGHT

    ## ---------------------------------------------------------------------##
    # Solve the inversion
    ## ---------------------------------------------------------------------##
    # Set gamma
    rf = 0.25

    # Load clusters
    clusters = xr.open_dataset(f'{data_dir}/clusters_permian.nc')['Clusters']

    # Identify the buffer cells
    buffer_cells = (np.unique(clusters, return_counts=True)[1] > 1)

    # Load the observations and observational difference
    y = np.load(f'{data_dir}/y_permian.npy').reshape((-1, 1))
    ya = np.load(f'{data_dir}/ya_permian.npy').reshape((-1, 1))
    y_diff = np.load(f'{data_dir}/y_diff_permian.npy').reshape((-1, 1))

    # Load Jacobian (ppb)
    K = np.load(f'{data_dir}/K_permian.npy')
    print(K.min(), K.max(), K.mean())

    # Create error vectors
    sa = 0.5**2*np.ones(K.shape[1])
    so = 15**2*np.ones(K.shape[0])/rf

    # Create prior
    xa = np.ones((K.shape[1], 1))

    # Calculate c
    c = ya - K @ xa

    # Calculate the gain matrix (/ppb) and the row-wise sum
    G = inv.get_gain_matrix(sa, so, K)
    # gsum = -G.sum(axis=1)
    
    # Calculate the posterior
    xhat = 1 + G @ y_diff

    # Calculate the contributions from the BC and from the emissions
    np.set_printoptions(precision=1, linewidth=300, suppress=True)
    tot_contrib = (G @ y_diff).reshape(-1,)[~buffer_cells]
    idx = np.argsort(tot_contrib)[::-1]
    tot_contrib = tot_contrib[idx]
    obs_contrib = (G @ y).reshape(-1,)[~buffer_cells][idx]
    BC_contrib = (G @ c).reshape(-1,)[~buffer_cells][idx]
    emis_contrib = (G @ (K @ xa)).reshape(-1,)[~buffer_cells][idx]
    gsum = 10*np.abs(G.sum(axis=1)[~buffer_cells][idx])
    print('Total :', tot_contrib)
    print('-'*30)
    print('y     :', obs_contrib)
    print('-'*30)
    print('bc    :', BC_contrib)
    print('-'*30)
    print('xa    :', emis_contrib)
    print('-'*30)
    fig, ax = fp.get_figax(rows=2)
    xp = np.arange(1, (~buffer_cells).sum() + 1)
    ax[0].plot(xp, gsum, color=fp.color(1), 
           lw=1, ls='--', label='Gain matrix sum')
    ax[0].plot(xp, tot_contrib, color=fp.color(3), lw=1,
            label='Total adjustment')
    ax[0].plot(xp, obs_contrib - BC_contrib, color=fp.color(5), lw=1,
            label='Observation - BC contribution')
    # ax[0].plot(xp, obs_contrib, color=fp.color(5), lw=1, ls='--',
    #         label='Observation contribution')
    # ax[0].bar(xp + 0.3, BC_contrib, bottom=tot_contrib + emis_contrib, 
    #        width=0.3, lw=1, ls='-', color=fp.color(7), 
    #        label='Boundary condition contribution')
    ax[0].bar(xp + 0.3, emis_contrib, bottom=tot_contrib,
           width=0.3, color=fp.color(7), lw=1, ls=':', 
           label='Emission contribution')

    ax[0] = fp.add_legend(ax[0], bbox_to_anchor=(1.1, 0.5), 
                          loc='center left', ncol=1, 
                          fontsize=ps.LABEL_FONTSIZE)
    ax[0].set_ylim(-0.1, 2)
    ax[0].set_xlim(0, len(xp)+1)

    norm = BC_contrib + emis_contrib
    ax[1].bar(xp, emis_contrib, bottom=0, color=fp.color(7), width=0.75,
              label='Relative emission contribution')
    ax[1].bar(xp, BC_contrib, bottom=emis_contrib,
              color=fp.color(5), 
              label='Relative BC contribution')
    # ax[1].set_ylim(0, 0.1)
    ax[1].set_xlim(0, len(xp)+1)

    ax1 = ax[1].twinx()
    ax1.plot(xp, gsum, color=fp.color(1), lw=1, ls='--', 
             label='Gain matrix sum')
    ax1.plot(xp, tot_contrib, color=fp.color(3), lw=1,
            label='Total adjustment')
    ax1.set_ylim(0, 2)

    ax[1] = fp.add_legend(ax[1], bbox_to_anchor=(1.1, 0.5), 
                  loc='center left', ncol=1, 
                  fontsize=ps.LABEL_FONTSIZE)

    
    # fig, ax = plot.format_plot(fig, ax, xp.max())
    fp.save_fig(fig, plot_dir, f'contribs')

    fig, ax = fp.get_figax(aspect=1)
    ax.scatter(gsum/10, emis_contrib/norm)
    fp.save_fig(fig, plot_dir, f'scatter')


#     ## ---------------------------------------------------------------------##
#     # Plot the gain matrix
#     ## ---------------------------------------------------------------------##
#     fig, ax = fp.get_figax()
#     ax.matshow(G, cmap='RdBu_r', vmin=-0.0001, vmax=0.0001)
#     fp.save_fig(fig, plot_dir, f'G_matrix')

#     ## ---------------------------------------------------------------------##
#     # Plot the orginal posterior
#     ## ---------------------------------------------------------------------##
#     xhat_cmap_1 = plt.cm.RdBu_r(np.linspace(0, 0.5, 256))
#     xhat_cmap_2 = plt.cm.RdBu_r(np.linspace(0.5, 1, 256))
#     xhat_cmap = np.vstack((xhat_cmap_1, xhat_cmap_2))
#     xhat_cmap = colors.LinearSegmentedColormap.from_list('xhat_cmap',
#                                                          xhat_cmap)
#     div_norm = colors.TwoSlopeNorm(vmin=0, vcenter=1, vmax=3.5)

#     xhat_cbar_kwargs = {'title' : r'Scale factor'}
#     xhat_kwargs = {'cmap' : xhat_cmap, 'norm' : div_norm,
#                    'vmin' : 0, 'vmax' : 3.5,
#                    'default_value' : 1,
#                    'cbar_kwargs' : xhat_cbar_kwargs,
#                    'map_kwargs' : small_map_kwargs}
#     fig, ax, c = plot.plot_state(xhat, clusters,
#                                  title='Posterior scale factors',
#                                  **xhat_kwargs)
#     fp.save_fig(fig, plot_dir, f'xhat')

#     ## ---------------------------------------------------------------------##
#     # Predict the magnitude of the BC error
#     ## ---------------------------------------------------------------------##
#     # Predict the magnitude of the BC error
#     BC_err = -gc.rma_modified(G.sum(axis=1)[buffer_cells], xhat[buffer_cells])
#     BC_err1 = -gc.rma(G.sum(axis=1)[buffer_cells], xhat[buffer_cells])

#     m, _, _, _ = np.linalg.lstsq(G.sum(axis=1)[buffer_cells].reshape((-1, 1)),
#                                  xhat[buffer_cells].reshape((-1, 1)),
#                                  rcond=None)
#     BC_err2 = -m[0][0]

#     m, _, _, _ = np.linalg.lstsq(xhat[buffer_cells].reshape((-1, 1)),
#                                  G.sum(axis=1)[buffer_cells].reshape((-1, 1)),
#                                  rcond=None)
#     BC_err3 = -1/m[0][0]

#     print('-'*70)
#     print(f'Predicted boundary condition error (RMA)         : {BC_err1:.1f}')
#     print(f'Predicted boundary condition error (RMA modified): {BC_err:.1f}')
#     print(f'Predicted boundary condition error (LSQ, x vs. g): {BC_err2:.1f}')
#     print(f'Predicted boundary condition error (LSQ, g vs. x): {BC_err3:.1f}')
#     print('-'*70)

#     # Plot scatter plot
#     fig, ax = fp.get_figax(aspect=1)
#     ax.scatter(xhat[buffer_cells], (G.sum(axis=1))[buffer_cells],
#                marker='v', s=20, color=fp.color(4), label='Buffer cells')
#     ax.scatter(xhat[~buffer_cells], (G.sum(axis=1))[~buffer_cells],
#                marker='x', s=20, color=fp.color(4), label='Non-buffer cells')
#     ax.plot(xhat, -xhat/BC_err, color=fp.color(4), ls='--')
#     ax = fp.add_labels(ax, r'$\hat{x}$', '$\Sigma$ G')
#     ax = fp.add_legend(ax, bbox_to_anchor=(0.5, -0.1), loc='upper center')
#     fp.save_fig(fig, plot_dir, 'scatter_orig')

#     # Plot scatter plot
#     fig, ax = fp.get_figax(aspect=1)
#     ax.scatter(xhat[buffer_cells], (-BC_err*G.sum(axis=1))[buffer_cells],
#                marker='v', s=20, color=fp.color(4), label='Buffer cells')
#     ax.scatter(xhat[~buffer_cells], (-BC_err*G.sum(axis=1))[~buffer_cells],
#                marker='x', s=20, color=fp.color(4), label='Non-buffer cells')

#     # ax.scatter(xhat[buffer_cells], (-BC_err2*G.sum(axis=1))[buffer_cells],
#     #            marker='v', s=20, color=fp.color(6), label='Buffer cells')
#     # ax.scatter(xhat[~buffer_cells], (-BC_err2*G.sum(axis=1))[~buffer_cells],
#     #            marker='x', s=20, color=fp.color(6), label='Non-buffer cells')

#     ax = fp.plot_one_to_one(ax)
#     ax = fp.add_labels(ax, r'$\hat{x}$', '$\Sigma$ G')
#     ax = fp.add_legend(ax, bbox_to_anchor=(0.5, -0.1), loc='upper center')
#     fp.save_fig(fig, plot_dir, 'scatter')

#     ## ---------------------------------------------------------------------##
#     # Method 1: Correct the posterior using Gsum
#     ## ---------------------------------------------------------------------##
#     # Plot G sum (absolute)
#     g_cmap_1 = plt.cm.RdBu_r(np.linspace(0, 0.5, 256))
#     g_cmap_2 = plt.cm.RdBu_r(np.linspace(0.5, 1, 256))
#     g_cmap = np.vstack((g_cmap_1, g_cmap_2))
#     g_cmap = colors.LinearSegmentedColormap.from_list('g_cmap', g_cmap)
#     g_div_norm = colors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=2)

#     gsum_cbar_kwargs = {'title' : r'Absolute scale factor error'}
#     gsum_kwargs = {'cmap' : g_cmap, 'norm' : g_div_norm,
#                    'vmin' : -1, 'vmax': 2,
#                    'default_value' : -1, 'cbar_kwargs' : gsum_cbar_kwargs,
#                    'map_kwargs' : small_map_kwargs}
#     fig, ax, c = plot.plot_state(-BC_err*G.sum(axis=1), clusters,
#                                  title=f'Absolute influence of\na {BC_err:.1f} ppb BC error',
#                                  **gsum_kwargs)
#     fp.save_fig(fig, plot_dir, f'gsum_abs')

#     # Plot corrected xhat
#     xhat_cbar_kwargs = {'title' : r'Corrected scale factor'}
#     xhat_kwargs = {'cmap' : xhat_cmap, 'norm' : div_norm,
#                    'vmin' : 0, 'vmax' : 3.5,
#                    'default_value' : 1,
#                    'cbar_kwargs' : xhat_cbar_kwargs,
#                    'map_kwargs' : small_map_kwargs}
#     fig, ax, c = plot.plot_state(xhat + BC_err*G.sum(axis=1), clusters,
#                                  title='Posterior scale factors',
#                                  **xhat_kwargs)
#     fp.save_fig(fig, plot_dir, f'xhat_corrected')

#     ## ---------------------------------------------------------------------##
#     # Method 2: Adjust So/Sa to allow buffer cells to better absorb errors
#     ## ---------------------------------------------------------------------##
#     # Plot G sum (relative)
#     gsum_cbar_kwargs = {'title' : r'Relative scale factor error'}
#     gsum_kwargs = {'cmap' : 'RdBu_r', 'vmin' : -1, 'vmax' : 1,
#                    'default_value' : 0,
#                    'cbar_kwargs' : gsum_cbar_kwargs,
#                    'map_kwargs' : small_map_kwargs}
#     fig, ax, c = plot.plot_state(-BC_err*G.sum(axis=1)/xhat, clusters,
#                                  title=f'Relative influence of\na {BC_err:.1f} ppb BC error',
#                                  **gsum_kwargs)
#     fp.save_fig(fig, plot_dir, f'gsum_rel')

#     # Step 1: ID observations that mainly influence the buffer grid cells
#     # K is    nstate 1    nstate 2     nstate 3      buffer cell
#     # obs 1
#     # obs 2
#     # obs 3
#     influence = K/K.sum(axis=1).reshape((-1, 1))
#     influence[:, ~buffer_cells] = 0
#     influence = influence.sum(axis=1)
#     influence = (influence >= 0.95)
#     # observations with 60% or moreinfluence onthe buffer cells
#     # The thresholds correspond to:
#     # 60% or more --> 71%  of observations
#     # 80% or more --> 60% of observations (need to check this)
#     # 90% or more --> 54% of observations
#     # 95% or more --> 49% of observations

#     # Now iterate through So and Sa
#     x = np.arange(-1, 1.01, 0.01)
#     sa_effect = np.zeros((len(x),))
#     so_effect = np.zeros((len(x),))
#     for ind, i in enumerate(x):
#         # Alter Sa
#         sa_alt = copy.deepcopy(sa)
#         sa_alt[buffer_cells] *= (10**(2*i))
#         g = inv.get_gain_matrix(sa_alt, so, K)
#         # ...need a measure of ILS

#         # Alter So
#         so_alt = copy.deepcopy(so)
#         so_alt[influence] *= (10**(2*i))
#         g = inv.get_gain_matrix()



# # # Create error vectors
# # sa = 0.5**2*np.ones(K.shape[1])
# # so = 15**2*np.ones(K.shape[0])/rf

# # # Calculate the gain matrix (/ppb) and the row-wise sum
# # G = inv.get_gain_matrix(sa, so, K)
# # gsum = -G.sum(axis=1)






#     # # Plot histograms
#     # fig, ax = fp.get_figax(aspect=4)
#     # ax.hist(xhat, histtype='step', bins=75, color=fp.color(2),
#     #         label='Posterior scale factors')
#     # ax.hist(gsum*BC_err, histtype='step', bins=150, color=fp.color(4),
#     #         label=f'Row-wise sum of G ({BC_err} ppb scaling)')
#     # ax.hist(gsum*BC_err/xhat, histtype='step', bins=150, color=fp.color(6),
#     #         label='Ratio')
#     # ax.set_xlim(0, 3)
#     # ax = fp.add_legend(ax, bbox_to_anchor=(0.5, -0.45), loc='upper center')
#     # ax = fp.add_labels(ax, 'Scale Factor', 'Count')
#     # fp.save_fig(fig, plot_dir, f'hist')


#     # ax.hist

#     # print(gsum*50)
#     # print('-'*20)
#     # print(xhat)
#     # print('-'*20)
#     # print(gsum*50/xhat)



