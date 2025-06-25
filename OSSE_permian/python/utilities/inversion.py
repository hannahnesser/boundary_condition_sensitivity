import numpy as np
import xarray as xr
import pandas as pd
from copy import deepcopy as dc
from sklearn.cluster import KMeans
from utilities import format_plots as fp
from utilities import utils, grid
from utilities import inversion_plot as ip

project_dir, config = utils.setup()
data_dir = f'{project_dir}/data/data_OSSE'

class OSSE:
    def __init__(self, xtrue, random_state):
        '''
        Assumes emission units of kg/km2/hr
        '''
        # Initiate the random state
        self.rs = np.random.RandomState(random_state)

        ## First, get the things that are the same for every Permian OSSE (as
        ## it is defined and executed here.)
        # Define clusters
        clusters = xr.open_dataarray(f'{data_dir}/StateVector.nc')
        clusters = clusters.where(clusters > 0, drop=True)

        # Get rid of the rightmost column because of some issue with the 
        # Jacobian
        clusters = clusters.where(clusters['lon'] != clusters['lon'].max(), 
                                  drop=True)
        self.keep_idx_state = np.sort((clusters.values - 1).flatten().astype(int))
        self.clusters = self.renumber_clusters(clusters)

        # Get the latitudes and longitudes of the individual observations
        self.lat = np.load(f'{data_dir}/lat_full.npy')
        self.lon = np.load(f'{data_dir}/lon_full.npy')
        # (We handle observations out of the domain later.)

        # And get the lats/lons of the super observations (these are averaged,
        # so we round to the nearest grid cell center)
        self.lat_super = np.round(np.load(f'{data_dir}/lat.npy')/0.25)*0.25
        self.lon_super = np.round(np.load(f'{data_dir}/lon.npy')/0.3125)*0.3125

        # Now get the index of observations to keep
        keep_idx_obs = (
            (self.lat_super >= self.clusters['lat'].min().values) &
            (self.lat_super <= self.clusters['lat'].max().values) &
            (self.lon_super >= self.clusters['lon'].min().values) &
            (self.lon_super <= self.clusters['lon'].max().values)
        )
        self.keep_idx_obs = np.arange(len(keep_idx_obs))[keep_idx_obs]
        self.lat_super = self.lat_super[self.keep_idx_obs]
        self.lon_super = self.lon_super[self.keep_idx_obs]

        # Define Jacobian and the resulting state vector/observational 
        # dimensions
        self.k_orig = np.load(f'{data_dir}/{config["k"]}')*1e9 # to ppb
        self.k_BC = self.k_orig[self.keep_idx_obs, :][:, -4:]
        self.k_orig = self.k_orig[self.keep_idx_obs, :][:, self.keep_idx_state]
        self.nobs = self.k_orig.shape[0]
        self.nstate = self.k_orig.shape[1]

        # Load the prior simulated values
        self.Fxa = np.load(f'{data_dir}/Fxa.npy')[self.keep_idx_obs].reshape(-1, 1)
        self.xa = np.append(np.ones(self.nstate), np.zeros(4)).reshape(-1, 1)
        self.ctrue = self.Fxa - np.append(self.k_orig, 
                                          self.k_BC, axis=1) @ self.xa

        # Define the emissions for which the Jacobian was constructed (EPA 
        # inventory, minus soil absorption, which we will assume is true) 
        # (and get the grid cell area along the way)
        xk = xr.open_dataset(f'{data_dir}/HEMCO_diagnostics.202005010000.nc')
        self.area = grid.clusters_2d_to_1d(xk['AREA'], self.clusters)*1e-6
        xk = grid.clusters_2d_to_1d(
            (xk['EmisCH4_Total'] - xk['EmisCH4_SoilAbsorb']).squeeze(drop=True), 
            self.clusters)
        xk *= (1e3)**2*(60*60) # kg/m2/s -> kg/km2/hr
        self.xk = xk
        
        # Define a default true emissions. If this isn't defined, we use the 
        # EDF inventory for 2019 supplemented by other sectors
        if xtrue is None:
            xother = xr.open_dataset(
                f'{data_dir}/HEMCO_diagnostics.202005010000.nc')
            xother = (
                xother['EmisCH4_Total'] - 
                xother['EmisCH4_Oil'] - 
                xother['EmisCH4_Gas'] - 
                xother['EmisCH4_SoilAbsorb']
            )
            xtrue = xr.open_dataset(f'{data_dir}/permian_EDF_2019.nc')
            xtrue = xtrue.sel(
                lat=clusters.where(self.clusters > 0, drop=True)['lat'],
                lon=clusters.where(self.clusters > 0, drop=True)['lon'])
            xtrue = xtrue['EmisCH4_Oil'] + xtrue['EmisCH4_Gas'] + xother
            xtrue = grid.clusters_2d_to_1d(xtrue, self.clusters)
            xtrue *= (1e3)**2*(60*60) # kg/m2/s --> kg/km2/hr
        elif len(xtrue) != self.nstate:
            raise ValueError('Provided xtrue is not of length nstate.')
        else:
            self.xtrue = xtrue
        self.xtrue = xtrue.reshape((-1, 1))

        # Generate the observations. The random errors on the observations  
        # were set by testing the mean variance in the true observations at 
        # 0.25 x 0.3125.
        self.y = (self.k_orig @ (self.xtrue/self.xk) + self.ctrue)

        # Get statistics of errors to apply to self.y (note that we are rounding 
        # floats that include fractional observations)
        self.count = np.load(f'{data_dir}/count.npy')[self.keep_idx_obs]
        yerr = np.array(
            [float(self.rs.normal(0, 10.5, np.ceil(cc).astype(int)).mean()/cc**0.5) 
             for cc in self.count]
        ).reshape(-1, 1)
        self.y = self.y + yerr

    @staticmethod
    def renumber_clusters(clusters):
        for i, idx in enumerate(np.unique(clusters)):
            clusters = clusters.where(clusters != idx, i + 1)
        return clusters

class Inversion(OSSE):
    def __init__(self, 
                 xa_abs=None, sa=None, so=None, BC_pert=0, gamma=1, xtrue=None, 
                 opt_BC=False, sa_BC=None, 
                 buffer=False, buffer_nrows=5, buffer_nclusters=None, 
                 buffer_p=1000,
                 random_state=config['random_state']):

        # Initialize the OSSE by inheriting from the parent class.
        OSSE.__init__(self, xtrue, random_state)

        # Adjust the Jacobian to be relative in terms of our actual prior
        if xa_abs is None:
            xa_abs = 3*self.xk.copy()
            # xa_abs = self.xtrue.copy() + self.rs.normal(0, 0.5, self.xtrue.shape)

        self.xa_abs = xa_abs.reshape((self.nstate, 1))
        self.k = self.k_orig * (self.xa_abs.T/self.xk.T)

        # Relative prior
        self.xa = np.ones((self.nstate, 1))

        # Prior errors
        if sa is None:
            sa = float(config['sa'])**2*np.ones(self.nstate)
        elif type(sa) in [float, int]:
            sa = sa**2*np.ones(self.nstate)
        elif len(sa) != self.nstate:
            raise ValueError('Provided prior errors are not of length nstate.')
        else:
            raise ValueError('Unknown condition for prior errors.')
        self.sa = sa.reshape((-1, 1))

        # Observing system errors
        if so is None:
            so = np.load(f'{data_dir}/so.npy')[self.keep_idx_obs]
        elif type(so) in [float, int]:
            so = so**2*np.ones(self.nobs)
        elif len(so) != self.nobs:
            raise ValueError('Provided observing system errors are not of'
                             ' length nobs.')
        else:
            print('Unknown condition for observing system errors.')
        self.so = so.reshape((-1, 1))

        # Adjust for buffer
        self.buffer = buffer
        if self.buffer:
            self.prepare_buffer_inversion(
                nrows=buffer_nrows, nclusters=buffer_nclusters, p=buffer_p)
        else:
            # Clip the domain if buffer is False (i.e., if we don't need room for
            # buffer grid cells). We do this now because we want to calculate c 
            # with the modified quantities. We'll take advantage of the existing
            # buffering algorithm with nclusters = 0.
            self.prepare_buffer_inversion(nrows=buffer_nrows, nclusters=0, p=1)

        # Adjust prior quantities for optimized BC
        self.opt_BC = opt_BC
        if self.opt_BC:
            self.sa_BC = sa_BC**2
            self.prepare_boundary_inversion()

        print('Inversion dimension: ')
        print(f'  State vector dimension: {self.nstate}')
        print(f'  Super-observation dimension: {self.nobs}')
        print(f'  Observation dimension: {len(self.lon)}')
        print(f'  Longitude limits: '
              f'{self.clusters["lon"].min().values - 0.3125/2}'
              f' - {self.clusters["lon"].max().values + 0.3125/2}')
        print(f'  Latitude limits: '
              f'{self.clusters["lat"].min().values - 0.25/2}'
              f' - {self.clusters["lat"].max().values + 0.25/2 }')

        # Set the boundary condition
        if type(BC_pert) in [float, int]:
            BC_pert = np.ones(4) * BC_pert
        self.c = self.ctrue + (self.k_BC @ BC_pert).reshape((-1, 1))

        # Get gamma/adjust for gamma
        if gamma is None:
            self.get_gamma()
        else:
            self.gamma = gamma
            self.so = self.so/self.gamma

        # Solve the inversion.
        self.solve_inversion()
        print('-'*70)


    def solve_inversion(self):
        # Solve the inversion
        kT_so_inv = self.k.T @ np.diag(1/self.so.reshape(-1,))
        sa_inv = np.diag(1/self.sa.reshape(-1,))

        self.shat = np.linalg.inv(sa_inv + kT_so_inv @ self.k)
        self.g = self.shat @ kT_so_inv
        self.a = np.identity(len(self.xa)) - self.shat @ sa_inv

        self.xhat = (self.xa + self.g @ (self.y - self.k @ self.xa - self.c))
        print('  Posterior range: ', self.xhat.min(), self.xhat.max())
        if self.opt_BC:
            print(f'  DOFS : {np.diag(self.a)[:-4].sum():.2f}')
        elif self.buffer_idx[0] != -1:
            print(f'  DOFS : {np.diag(self.a)[self.cluster_idx].sum():.2f}')
        else:
            print(f'  DOFS : {np.trace(self.a):.2f}')


    def cost_prior(self):
        return ((self.xhat - self.xa)**2/self.sa).sum()
    

    def cost_obs(self):
        return ((self.y - self.ya)**2/self.so).sum()


    def get_gamma(self, tol=1e-1):
        print('Finding gamma...')
        gamma = 10
        gamma_not_found = True
        so_orig = dc(self.so)
        while gamma_not_found:
            self.so = so_orig/gamma
            self.solve_inversion()
            cost = self.cost_prior()/self.nstate
            print(f'{gamma:.2f}: {cost:.3f}')
            if np.abs(cost - 1) <= tol:
                gamma_not_found = False
            elif cost > 1:
                gamma /= 2
            elif cost < 1:
                gamma *= 1.5
        print('Gamma found!')
        self.gamma = gamma


    def prepare_boundary_inversion(self):
        self.k = np.append(self.k, self.k_BC, axis=1)
        self.sa = np.append(self.sa, self.sa_BC*np.ones(4))
        self.xa = np.append(self.xa.flatten(), np.zeros(4))[:, None]
        self.xa_abs = np.append(self.xa_abs.flatten(), np.ones(4))[:, None]
        self.nstate += 4


    def remove_boundary_inversion(self):
        self.k = self.k[:, :-4]
        self.sa = self.sa[:-4]
        self.xa = self.xa[:-4][:, None]
        self.xa_abs = self.xa_abs[:-4][:, None]
        self.nstate -= 4


    def prepare_buffer_inversion(self, nrows, nclusters, p):
        # Generate the new cluster file that includes buffered grid cells. Also
        # get the list of which grid cells are considered "buffered."
        clusters_new, self.buffer_idx = self._buffer_clusters(nrows, nclusters)

        # Get the cluster indexes
        cluster_idx = clusters_new.values.flatten()
        cluster_idx = cluster_idx[~np.isnan(cluster_idx)].astype(int) - 1
        cluster_idx = np.unique(cluster_idx)
        self.cluster_idx = cluster_idx[~np.isin(cluster_idx, self.buffer_idx)]

        # Generate a mapping between the two
        map = xr.Dataset({'new' : clusters_new, 'old' : self.clusters})
        map = map.to_dataframe().sort_values('old').reset_index(drop=True)
        self.clusters = clusters_new

        # Now, we need to modify the inversion components.
        self.nstate = int(self.clusters.max())

        # xa: Non-BC elements are all 1 but are now length nstate_new
        self.xa = np.ones((self.nstate))[:, None]

        # xa_abs: Need to add in mass/time space, then divide by new area
        xa_abs = pd.DataFrame({'old' : map['old'],
                               'clusters' : map['new'],
                               'xa_abs' : (self.xa_abs * self.area).flatten(),
                               'xk' : (self.xk * self.area).flatten(),
                               'xtrue' : (self.xtrue * self.area).flatten(),
                               'area' : self.area.flatten()})
        xa_abs = xa_abs.groupby('clusters').sum()
        xa_abs['xa_abs'] = xa_abs['xa_abs']/xa_abs['area']
        xa_abs['xk'] = xa_abs['xk']/xa_abs['area']
        xa_abs['xtrue'] = xa_abs['xtrue']/xa_abs['area']
        self.xa_abs_full = self.xa_abs.copy()
        self.xtrue_full = self.xtrue.copy()
        self.area = xa_abs['area'].values
        self.xa_abs = xa_abs['xa_abs'].values
        self.xk = xa_abs['xk'].values
        self.xtrue = xa_abs['xtrue'].values

        # sa: Non-buffer elements are the same, buffer elements are scaled by p
        sa = float(config['sa'])**2*np.ones(self.nstate)
        sa[self.buffer_idx] = p**2 * sa[self.buffer_idx]
        self.sa = sa.reshape((-1, 1))

        # k: As long as k is scaled by the emissions (e.g., relative xa), add
        #    together the buffer columns
        k = pd.DataFrame(self.k.T)
        k['clusters'] = map['new']
        self.k = k.groupby('clusters').sum().T.values

        k_orig = pd.DataFrame(self.k_orig.T)
        k_orig['clusters'] = map['new']
        self.k_orig = k_orig.groupby('clusters').sum().T.values

        # so, y, ya: No changes unless nclusters == 0
        if nclusters == 0:
            self.cluster_idx = self.cluster_idx[1:]
            subset = self.clusters.where(self.clusters > 0, drop=True)
            subset_idx_obs = (
                (self.lat_super >= subset['lat'].min().values) &
                (self.lat_super <= subset['lat'].max().values) &
                (self.lon_super >= subset['lon'].min().values) &
                (self.lon_super <= subset['lon'].max().values)
            )
            self.xa_abs = self.xa_abs[1:]
            self.xk = self.xk[1:]
            self.xtrue = self.xtrue[1:]
            self.area = self.area[1:]
            self.k = self.k[subset_idx_obs, :][:, 1:] # We remove the 0th index
            self.k_orig = self.k_orig[subset_idx_obs, :][:, 1:]
            self.k_BC = self.k_BC[subset_idx_obs, :]
            self.so = self.so[subset_idx_obs]
            self.Fxa = self.Fxa[subset_idx_obs]
            self.y = self.y[subset_idx_obs]
            self.nobs = subset_idx_obs.sum()
            self.count = self.count[subset_idx_obs]
            self.lat_super = self.lat_super[subset_idx_obs]
            self.lon_super = self.lon_super[subset_idx_obs]

            # Recalculate ctrue
            print(np.append(self.k_orig, self.k_BC, axis=1).shape)
            kx = (np.append(self.k_orig, self.k_BC, axis=1) @ 
                  np.append(self.xa, np.zeros(4)))[:, None]
            self.ctrue = self.Fxa - kx


    def _buffer_clusters(self, nrows, nclusters):
        # If 0 clusters, we'll actually use nclusters = 1 and then remove it
        # later
        if nclusters == 0:
            nclusters_orig = 0
            nclusters = 1
        else: 
            nclusters_orig = nclusters

        # First, select the grid cells that will remain unchanged.
        c = self.clusters.copy()
        clusters_new = c.where((c['lat'] <= c['lat'][-(nrows + 1)]) &
                               (c['lat'] >= c['lat'][nrows]) &
                               (c['lon'] <= c['lon'][-(nrows + 1)]) &
                               (c['lon'] >= c['lon'][nrows]))
        
        # Get the old index as a record, and then renumber the new clusters
        old_idx = clusters_new.values.flatten()
        old_idx = old_idx[~np.isnan(old_idx)]
        clusters_new = self.renumber_clusters(clusters_new)
        
        # Select the buffer grid cells in the nrows surrounding the domain
        buffer_idx = c.where(~c.isin(old_idx))

        # If nclusters is specified, cluster these buffer grid cells using
        # Kmeans clustering
        if nclusters is not None:
            # Clustering
            buffer_labels = buffer_idx.to_dataframe().dropna().reset_index()
            kmeans = KMeans(n_clusters=nclusters, random_state=0, n_init='auto')
            buffer_labels['StateVector'] = kmeans.fit_predict(
                buffer_labels[['lat', 'lon']])

            # These labels are now 0 - nclusters. Adjust accordingly
            buffer_labels['StateVector'] += 1 + clusters_new.max().values

            # And convert it to a dataset
            buffer_idx = xr.DataArray.from_series(
                buffer_labels.set_index(['lat', 'lon'])['StateVector'])
        # Otherwise, just update the labels
        else:
            buffer_idx = self.renumber_clusters(buffer_idx) + clusters_new.max()

        if nclusters != nclusters_orig:
            print('Triggering 0 clustering.')
            buffer_idx = buffer_idx - buffer_idx.max()

        # Now combine the two. Save out the buffer index so that we know which 
        # state vector elements to set to large prior errors.
        clusters_new = clusters_new.fillna(0) + buffer_idx.fillna(0)
        buffer_idx = buffer_idx.values.flatten()
        buffer_idx = buffer_idx[~np.isnan(buffer_idx)]
        buffer_idx = np.unique(buffer_idx).astype(int) - 1

        # Return the new clusters and the buffer labels
        return clusters_new, buffer_idx


    def preview(self, sa_bc, ils_threshold=0.1, 
                lat_delta=0.25, lon_delta=0.3125,
                plot_dir=None, plot_str=None):
        if plot_dir is not None:
            fig, ax = fp.get_figax(cols=2, rows=2, maps=True, 
                                   lats=self.clusters.lat, 
                                   lons=self.clusters.lon)
            fig.subplots_adjust(hspace=1)

        g = self.clusters.copy().to_dataset()
        g['x_i'] = grid.clusters_1d_to_2d(self.xa_abs, self.clusters)

        # Get the minimum distance to the edge for each grid point. 
        # D1: Distance to the southern border
        # D2: Distance to the northern border
        # D3: Distance to the western border
        # D4: Distance to the eastern border
        g = g.where(g['StateVector'] > 0, 
                    drop=True).to_dataframe().reset_index()
        g['D1'] = grid.distance(g['lon'], g['lat'], 
                                g['lon'], g['lat'].min() - lat_delta)
        g['D2'] = grid.distance(g['lon'], g['lat'], 
                                g['lon'], g['lat'].max() + lat_delta)
        g['D3'] = grid.distance(g['lon'], g['lat'], 
                                g['lon'].min() - lon_delta, g['lat'])
        g['D4'] = grid.distance(g['lon'], g['lat'], 
                                g['lon'].max() + lon_delta, g['lat'])
        D = g[['D1', 'D2', 'D3', 'D4']].min(axis=1).values
        D_idx = g[['D1', 'D2', 'D3', 'D4']].values.argmin(axis=1) # km

        if plot_dir is not None:
            fig, ax[0, 0], c = ip.plot_state(
                g['x_i'], self.clusters, title=r'Prior emissions',
                vmin=0, vmax=6,
                fig_kwargs={'figax' : [fig, ax[0, 0]]},
                cbar_kwargs={'horizontal' : True, 
                             'title' : 'Methane emissions\n'r'(kg/km$^2$/hr)'})

            fig, ax[0, 1], c = ip.plot_state(
                D, self.clusters, title=r'Distance to boundary ($L_{up}$)',
                fig_kwargs={'figax' : [fig, ax[0, 1]]},
                cbar_kwargs={'horizontal' : True, 'title' : 'Distance (m)'})

        # Get the corresponding upstream emissions (taking the mean of xa_abs 
        # technically doesn't account for grid cell size, but this should be
        # fine on regional domains)
        # First, get observation counts for each grid cell (needed for upstream
        # counts)
        lats = np.round(self.lat/lat_delta)*lat_delta
        lons = np.round(self.lon/lon_delta)*lon_delta
        counts = pd.DataFrame({'lat' : lats,
                               'lon' : lons,
                               'count_i' : np.ones(len(self.lat))})
        counts = counts.groupby(['lat', 'lon']).count()['count_i'].reset_index()
        g = pd.merge(g, counts, on=['lat', 'lon'], how='left')
        g['count_i'] = g['count_i'].fillna(0)

        g['x_up'] = np.ones(D.shape) # kg/km2/hr
        g['count_up'] = np.zeros(D.shape)

        def get_upstream_info(var, g, lat_or_lon, idx, D_idx=D_idx):
            lats_or_lons = np.sort(np.unique(g[D_idx == idx][lat_or_lon]))
            cond = (D_idx == idx) & (g[lat_or_lon].isin(lats_or_lons))
            # oper = ('mean' if var == 'x' else 'sum')
            var_name = ('x_i' if var == 'x' else 'count_i')
            up = g[cond].groupby(lat_or_lon)[var_name].agg(['median'])['median']
            if idx in [1, 3]:
                up = up[::-1]
            up = np.append(0, np.cumsum(up).values[:-1]) + 0*up
            g.loc[cond, f'{var}_up'] = up.loc[
                g.loc[cond, lat_or_lon].values].values

        for ll, ii in zip(['lat', 'lat', 'lon', 'lon'], [0, 1, 2, 3]):
            get_upstream_info('x', g, ll, ii)
            get_upstream_info('count', g, ll, ii)

        if plot_dir is not None:
            fig, ax[1, 0], c = ip.plot_state(
                g['x_up'], self.clusters,
                title=r'Upstream emissions ($x_{up}$)', 
                fig_kwargs={'figax' : [fig, ax[1, 0]]},
                cbar_kwargs={'horizontal' : True, 
                             'title' : 'Methane emissions\n'r'(kg/km$^2$/hr)'})
            fig, ax[1, 1], c = ip.plot_state(
                g['count_up'], self.clusters,
                title=r'Upstream observation count', 
                fig_kwargs={'figax' : [fig, ax[1, 1]]},
                cbar_kwargs={'horizontal' : True, 
                             'title' : 'Count'})
            fp.save_fig(fig, plot_dir, plot_str)

        # Get the size of each grid cell (approximate)
        g['lat_min'] = g['lat'] - lat_delta/2
        g['lat_max'] = g['lat'] + lat_delta/2
        g['lon_min'] = g['lon'] - lon_delta/2
        g['lon_max'] = g['lon'] + lon_delta/2
        g['lat_dist'] = grid.distance(g['lon'], g['lat_min'], 
                                      g['lon'], g['lat_max'])
        g['lon_dist'] = grid.distance(g['lon_min'], g['lat'],
                                      g['lon_max'], g['lat'])
        L = np.sqrt(g['lat_dist']*g['lon_dist']) # km

        # Get masks for where to calculate the first grid cell approx vs.
        # the standard expression
        outer_row = (D <= L*1.5)

        # Assume some basic parameters for the estimation of k
        U = 5*(60**2/1000) # Wind speed, m/s -> km/hr
        Mair = 28.97 # Molar mass dry air, g/mol
        MCH4 = 16.01 # Molar mass methane, g/mol # These units cancel out
        grav = 9.8/1000*(60**4) # Acceleration due to gravity, m/s2 -> km/hr2
        p = 1e5*1000*(60**4) # Surface pressure, Pa = kg/m/s2 -> kg/km/hr2
        k_i = 1e9*(Mair/MCH4)*L*grav/(U*p)
        k_up = 1e9*(Mair/MCH4)*D*grav/(U*p)
        # ()(km)(km/hr2)(hr/km)(hr2 km/kg) = (ppb km2 hr/kg)

        # Get the uncertainties (we're going to do a messy averaging because
        # we really know that it's the same everywhere in our system)
        g['so_i'] = 15**2/g['count_i']/self.gamma
        g['so_up'] = 15**2/g['count_up']/self.gamma

        # If there are no observations, set the observing errors to be very 
        # large. This should send delta xhat to be 0 in those grid cells...
        # which is sort of odd, but it reflects the fact that there was no 
        # constraint in the original inversion, and that doesn't change
        g[['so_i', 'so_up']] = g[['so_i', 'so_up']].fillna(40**2)
        g[['so_i', 'so_up']] = g[['so_i', 'so_up']].replace([np.inf], 40**2)

        # Reshape sa
        sa_i = self.sa.reshape(-1,)*g['x_i']**2 # abs
        sa_up = self.sa.reshape(-1,)*g['x_up']**2
        if type(sa_bc) not in [float, int]:
            sa_bc = sa_bc.max()

        R_i = g['so_i']/(k_i**2*sa_i)
        R_up = g['so_up']/(k_up**2*sa_up)
        j = D/L
        beta = j**2*sa_up/sa_i + 1
        diff = - (sa_bc/k_i)*R_up/(R_up*R_i + R_i + beta*R_up + 1)
        diff_0 = - (sa_bc/k_i)/(1 + R_i)
        diff[outer_row] = diff_0[outer_row]

        # While we're at it, we'll estimate an influence length scale using
        # the 1D approximation
        n_1d = int(self.nstate**0.5)
        so_mean = (np.mean(g['so_i']**0.5)**2)
        sa = np.mean(self.sa**0.5)**2*np.mean(self.xa_abs)**2

        print('Estimating R')
        print(f'  Mean K : {np.mean(k_i):.2f}')
        print(f'  Mean sa : {sa**0.5:.2f}')
        print(f'  Mean so : {so_mean**0.5}')
        print(f'  R : {so_mean/(np.mean(k_i)**2*sa):.2f}')

        return (diff/g['x_i'])[::-1]