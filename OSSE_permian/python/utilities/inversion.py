import numpy as np
import xarray as xr
import pandas as pd
from copy import deepcopy as dc
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

        # Define Jacobian and the resulting state vector/observational 
        # dimensions
        self.k = np.load(f'{data_dir}/{config["k"]}')*1e9 # to ppb
        self.nobs = self.k.shape[0]
        self.nstate = self.k.shape[1] - 4
        print('Inversion dimension: ')
        print(f'  State vector dimension: {self.nstate}')
        print(f'  Observation dimension: {self.nobs}')
        # print(f'  {self.k[:, :-4] @ xa[:-4, :]}')

        # Get the latitudes and longitudes of the observations
        self.lat = np.load(f'{data_dir}/lat_full.npy')
        self.lon = np.load(f'{data_dir}/lon_full.npy')

        # Load the prior simulated values
        self.Fxa = np.load(f'{data_dir}/Fxa.npy').reshape(-1, 1)
        self.xa = np.append(np.ones(self.nstate), np.zeros(4)).reshape(-1, 1)
        self.ctrue = self.Fxa - self.k @ self.xa
        
        # Define the emissions for which the Jacobian was constructed (EPA 
        # inventory, minus soil absorption, which we will assume is true) 
        # (and get the grid cell area along the way)
        xk = xr.open_dataset(f'{data_dir}/HEMCO_diagnostics.202005010000.nc')
        self.area = grid.clusters_2d_to_1d(xk['AREA'], clusters)
        xk = grid.clusters_2d_to_1d(
            (xk['EmisCH4_Total'] - xk['EmisCH4_SoilAbsorb']).squeeze(drop=True), 
            clusters)
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
                lat=clusters.where(clusters > 0, drop=True)['lat'],
                lon=clusters.where(clusters > 0, drop=True)['lon'])
            xtrue = xtrue['EmisCH4_Oil'] + xtrue['EmisCH4_Gas'] + xother
            xtrue = grid.clusters_2d_to_1d(xtrue, clusters)
            xtrue *= (1e3)**2*(60*60) # kg/m2/s --> kg/km2/hr
        elif len(xtrue) != self.nstate:
            raise ValueError('Provided xtrue is not of length nstate.')
        else:
            self.xtrue = xtrue
        self.xtrue = xtrue.reshape((-1, 1))

        # # Define the true boundary condition
        # if (type(BCtrue) not in [float, int]):
        #     if  len(BCtrue) != self.nobs:
        #         raise ValueError('The provided true boundary condition is not '
        #                          'of length 1 or nobs.')
        # self.ctrue = BCtrue*np.ones((self.nobs, 1))

        # Generate the observations. The random errors on the observations  
        # were set by testing the mean variance in the true observations at 
        # 0.25 x 0.3125.
        self.y = (self.k[:, :-4] @ (self.xtrue/self.xk) + self.ctrue)
                #   + self.rs.normal(0, 8, (self.nobs, 1)))
        
        # Get statistics of errors to apply to self.y (note that we are rounding 
        # floats that include fractional observations)
        self.count = np.load(f'{data_dir}/count.npy')
        yerr = np.array(
            [float(self.rs.normal(0, 10.5, np.ceil(c).astype(int)).mean()/c**0.5) 
             for c in self.count]
        ).reshape(-1, 1)
        self.y = self.y + yerr


    def calculate_total_emissions(self, mask):
        ...



class Inversion(OSSE):
    def __init__(self, xa_abs=None, sa=None, so=None, BC_pert=0, gamma=1,
                 xtrue=None, opt_BC=False,
                 random_state=config['random_state']):
        
        ## Initialize the OSSE by inheriting from the parent class
        OSSE.__init__(self, xtrue, random_state)

        # Shorten things if the boundary condition isn't optimized
        if ~opt_BC:
            self.k = self.k[:, :-4]

        # Set the boundary condition
        self.c = self.ctrue + BC_pert

        # Absolute prior: if undefined, just use a prior that is relatively
        # flat with large enhancements only where the EPA inventory is above
        # its average. This was tested and produced reasonable inverse 
        # results in combination with the true BC.
        if xa_abs is None:
            # xa_abs = 0.1*np.ones((self.nstate, 1))
            # xa_abs[self.xk > self.xk.mean()] = 3
            xa_abs = self.xk.copy()
        self.xa_abs = xa_abs.reshape((self.nstate, 1))
        
        # Adjust the Jacobian to be relative in terms of our actual prior
        self.k = self.k * (self.xa_abs.T/self.xk.T)
        # rat = self.xa_abs.T/self.xk.T
        # print('Ratio : ', rat.min(), rat.max())
        # print('xa range: ', self.xa_abs.min(), self.xa_abs.max(), self.xa_abs.mean())
        # print('xk range: ', self.xk.min(), self.xk.max(), self.xk.mean())

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
            so = np.load(f'{data_dir}/so.npy')
        elif type(so) in [float, int]:
            so = so**2*np.ones(self.nobs)
        elif len(so) != self.nobs:
            raise ValueError('Provided observing system errors are not of'
                             ' length nobs.')
        else:
            print('Unknown condition for observing system errors.')
        self.so = so.reshape((-1, 1))

        if gamma is None:
            self.get_gamma()
        else:
            self.gamma = gamma

        # Solve the inversion.
        self.solve_inversion()


    def solve_inversion(self):
        # Solve the inversion
        so = self.so/self.gamma
        kso_inv = self.k/so
        sa_inv = np.diag(1/self.sa.reshape(-1,))

        self.shat = np.linalg.inv(sa_inv + kso_inv.T @ self.k)
        self.g = self.shat @ kso_inv.T
        self.a = np.identity(len(self.xa)) - self.shat @ sa_inv
        self.xhat = (self.xa + self.g @ (self.y - self.k @ self.xa - self.c))
        print('  Posterior range: ', self.xhat.min(), self.xhat.max())


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
        print('-'*70)

        self.gamma = gamma
        self.so = self.so/self.gamma

    def estimate_delta_xhat(self, clusters, sa_bc, 
                            lat_delta=0.25, lon_delta=0.3125,
                            plot_dir=None, plot_str=None):
        if plot_dir is not None:
            fig, ax = fp.get_figax(cols=2, rows=2, maps=True, 
                                   lats=clusters.lat, lons=clusters.lon)
            fig.subplots_adjust(hspace=1)

        g = clusters.copy().to_dataset()
        g['x_i'] = grid.clusters_1d_to_2d(self.xa_abs, clusters)

        # Get the minimum distance to the edge for each grid point. 
        # D1: Distance to the southern border
        # D2: Distance to the northern border
        # D3: Distance to the western border
        # D4: Distance to the eastern border
        g = g.where(g['StateVector'] > 0, 
                    drop=True).to_dataframe().reset_index()
        g['D1'] = grid.distance(g['lon'], g['lat'], 
                                g['lon'], g['lat'].min() - lat_delta/2)
        g['D2'] = grid.distance(g['lon'], g['lat'], 
                                g['lon'], g['lat'].max() + lat_delta/2)
        g['D3'] = grid.distance(g['lon'], g['lat'], 
                                g['lon'].min() - lon_delta/2, g['lat'])
        g['D4'] = grid.distance(g['lon'], g['lat'], 
                                g['lon'].max() + lon_delta/2, g['lat'])
        D = g[['D1', 'D2', 'D3', 'D4']].min(axis=1).values
        D_idx = g[['D1', 'D2', 'D3', 'D4']].values.argmin(axis=1) # km

        if plot_dir is not None:
            fig, ax[0, 0], c = ip.plot_state(
                g['x_i'], clusters, title=r'Prior emissions',
                vmin=0, vmax=6,
                fig_kwargs={'figax' : [fig, ax[0, 0]]},
                cbar_kwargs={'horizontal' : True, 
                             'title' : 'Methane emissions\n'r'(kg/km$^2$/hr)'})

            fig, ax[0, 1], c = ip.plot_state(
                D, clusters, title=r'Distance to boundary ($L_{up}$)',
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
            up = np.append(0, np.cumsum(up))[:-1] + up/2
            g.loc[cond, f'{var}_up'] = up.loc[g.loc[cond, 
                                                    lat_or_lon].values].values

        for ll, ii in zip(['lat', 'lat', 'lon', 'lon'], [0, 1, 2, 3]):
            get_upstream_info('x', g, ll, ii)
            get_upstream_info('count', g, ll, ii)
        
        if plot_dir is not None:
            fig, ax[1, 0], c = ip.plot_state(
                g['x_up'], clusters,
                title=r'Upstream emissions ($x_{up}$)', 
                fig_kwargs={'figax' : [fig, ax[1, 0]]},
                cbar_kwargs={'horizontal' : True, 
                             'title' : 'Methane emissions\n'r'(kg/km$^2$/hr)'})
            fig, ax[1, 1], c = ip.plot_state(
                g['count_up'], clusters,
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
        
        # Assume some basic parameters for the estimation of k
        U = 4*(60**2/1000) # Wind speed, m/s -> km/hr
        Mair = 28.97 # Molar mass dry air, g/mol
        MCH4 = 16.01 # Molar mass methane, g/mol # These units cancel out
        grav = 9.8/1000*(60**4) # Acceleration due to gravity, m/s2 -> km/hr2
        p = 1e5*1000*(60**4) # Surface pressure, Pa = kg/m/s2 -> kg/km/hr2

        # Calculate k_est
        k_i = 1e9*(Mair/MCH4)*L*grav/(U*p)
        k_up = 1e9*(Mair/MCH4)*L*grav/(U*p)
        # ()(km)(km/hr2)(hr/km)(hr2 km/kg) = (ppb km2 hr/kg)

        # Get the uncertainties (we're going to do a messy averaging because
        # we really know that it's the same everywhere in our system)
        # so_i = pd.DataFrame({'lat' : lats, 'lon' : lons, 
        #                      'so' : 15**2*np.ones(len(lats))})
        # so_i = so_i.groupby(['lat', 'lon'], observed=False).agg(['mean', 'count'])
        # so_i = (((so_i['so']['mean']/so_i['so']['count'])**0.5)**2).reset_index() #ppb^2
        # so_i = so_i.rename(columns={0 : 'so'})
        g['so_i'] = 15**2/g['count_i']
        g['so_up'] = 15**2/g['count_up']

        # If there are no observations, set the observing errors to be very 
        # large. This should send delta xhat to be 0 in those grid cells...
        # which is sort of odd, but it reflects the fact that there was no 
        # constraint in the original inversion, and that doesn't change
        g[['so_i', 'so_up']] = g[['so_i', 'so_up']].fillna(40**2)
        # so['so'] = so['so'].fillna(40**2)
        # g = pd.merge(g, so, on=['lat', 'lon'])

        # Reshape sa
        sa = self.sa.reshape(-1,) # rel

        # Calculate delta xhat
        # print(kL.min(), sa.min(), g['xa_abs'].min())
        # print(k_up.min(), g['x_up'].min(), g['so'].min())
        num = k_i * sa * g['x_i'] * sa_bc # (ppb km2 hr/kg)()(kg/km2/hr)(ppb) = ppb^2
        den = sa*((k_up*g['x_up'])**2 + (k_i*g['x_i'])**2) + g['so_i'] # ppb^2
        delta_xhat = - num / den

        # Calculate the more complicated (2x2 model) delta xhat
        rat = sa / g['so_i'] # 1 / ppb
        # n = L/
        num = L * sa_bc * k_i * sa * g['so_i'] * g['x_i']
        # km ppb ppb km2 hr /kg ppb2 kg /km2 /hr = km ppb4
        den1 = D * k_up**2 * k_i**2 * sa**2 * g['x_i']**2 * g['x_up']**2
        # km ppb2 km4 hr2 /kg2 ppb2 km4 hr2 /kg2 kg2 /km4 /hr2 kg2 /km4 /hr2 =  km ppb4
        den2 = (D + L) * k_up**2 * sa * g['so_i'] * g['x_up']**2
        # km ppb2 /flux2 ppb2 flux2 = km ppb4
        den3 = L * k_i**2 * sa * g['so_i'] * g['x_i']**2
        # km ppb2 /flux2 ppb2 flux2 = km ppb4
        den4 = L * g['so_i']**2
        # km ppb4
        delta_xhat_2x2 = - num / (den1 + den2 + den3 + den4)
        approx = - num / den4

        # Yet another attempt
        kx_i = k_i * g['x_i']
        kx_up = k_up * g['x_up']
        s_a_up = sa / g['so_up']
        s_i_up = g['so_i'] / g['so_up']
        s_i_a = g['so_i'] / sa

        print(sa_bc)
        print(kx_i.min()**2, kx_i.max()**2, (kx_i**2).mean())
        print(kx_up.min()**2, kx_up.max()**2, (kx_up**2).mean())
        num = (sa_bc * kx_i)
        den1 = kx_up**2 * s_a_up * kx_i**2
        den2 = kx_up**2 * s_i_up
        # den2 = s_i_up
        den3 = kx_up**2 
        den4 = kx_i**2
        den5 = s_i_a
        delta_xhat = - num / (den1 + den2 + den3 + den4 + den5)

        return delta_xhat.values[::-1], delta_xhat_2x2.values[::-1], g, num[::-1], den1[::-1], den2[::-1], den3[::-1], den4[::-1], den5[::-1]
