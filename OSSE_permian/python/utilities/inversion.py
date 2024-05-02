import numpy as np
import xarray as xr
from copy import deepcopy as dc
from utilities import format_plots as fp
from utilities import utils, grid

project_dir, config = utils.setup()
data_dir = f'{project_dir}/data/OSSE'

class OSSE:
    def __init__(self, xtrue, BCtrue, random_state):
        '''
        Assumes emission units of kg/km2/hr
        '''
        # Initiate the random state
        self.rs = np.random.RandomState(random_state)

        ## First, get the things that are the same for every Permian OSSE (as
        ## it is defined and executed here.)
        # Define clusters
        clusters = xr.open_dataarray(f'{data_dir}/StateVector.nc')

        # Define Jacobian and the resulting state vector/observational 
        # dimensions
        self.k = np.load(f'{data_dir}/{config["k"]}')*1e9 # to ppb
        self.nobs = self.k.shape[0]
        self.nstate = self.k.shape[1]

        # Get the latitudes and longitudes of the observations
        self.lat = np.load(f'{data_dir}/lat.npy')
        self.lon = np.load(f'{data_dir}/lon.npy')
        
        # Define the emissions for which the Jacobian was constructed (EPA 
        # inventory, minus soil absorption, which we will assume is true) 
        # (and get the grid cell area along the way)
        xk = xr.open_dataset(f'{data_dir}/HEMCO_diagnostics.202005010000.nc')
        self.area = grid.clusters_2d_to_1d(xk['AREA'], clusters)
        # xk = xk['EmisCH4_Total'].squeeze(drop=True)
        xk = grid.clusters_2d_to_1d(
            xk['EmisCH4_Total'].squeeze(drop=True), clusters)
        xk *= (1e3)**2*(60*60) # -> kg/km2/hr
        self.xk = xk
        
        # Define a default true emissions. If this isn't defined, we use the 
        # EDF inventory for 2019.
        if xtrue is None:
            xtrue = xr.open_dataset(f'{data_dir}/permian_EDF_2019.nc')
            xtrue = xtrue.sel(
                lat=clusters.where(clusters > 0, drop=True)['lat'],
                lon=clusters.where(clusters > 0, drop=True)['lon'])
            xtrue = xtrue['EmisCH4_Oil'] + xtrue['EmisCH4_Gas']
            xtrue = grid.clusters_2d_to_1d(xtrue, clusters)
            xtrue *= (1e3)**2*(60*60) # kg/m2/s --> kg/km2/hr
        elif len(xtrue) != self.nstate:
            raise ValueError('Provided xtrue is not of length nstate.')
        else:
            self.xtrue = xtrue
        self.xtrue = xtrue.reshape((-1, 1))
        
        # Define the true boundary condition
        if (type(BCtrue) not in [float, int]):
            if  len(BCtrue) != self.nobs:
                raise ValueError('The provided true boundary condition is not '
                                 'of length 1 or nobs.')
        self.ctrue = BCtrue*np.ones((self.nobs, 1))

        # Generate the observations. The random errors on the observations  
        # were set by testing the mean variance in the true observations at 
        # 0.25 x 0.3125.
        self.y = (self.k @ (self.xtrue/self.xk) + self.ctrue 
                  + self.rs.normal(0, 10, (self.nobs, 1)))
    

    def calculate_total_emissions(self, mask):
        ...



class Inversion(OSSE):
    def __init__(self, xa_abs=None, sa=None, so=None, BC=None,
                 xtrue=None, BCtrue=config['bc_true'],
                 random_state=config['random_state']):
        
        ## Initialize the OSSE by inheriting from the parent class
        OSSE.__init__(self, xtrue, BCtrue, random_state)

        # Absolute prior: if undefined, just use a prior that is relatively
        # flat with large enhancements only where the EPA inventory is above
        # its average. This was tested and produced reasonable inverse 
        # results in combination with the true BC.
        if xa_abs is None:
            xa_abs = 0.5*np.ones((self.nstate, 1))
            xa_abs[self.xk > self.xk.mean()] = 3
        self.xa_abs = xa_abs.reshape((self.nstate, 1))
        
        # Adjust the Jacobian to be relative in terms of our actual prior
        self.k = self.k * (self.xa_abs.T/self.xk.T)
        
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
            so = float(config['so'])**2*np.ones(self.nobs)
        elif type(so) in [float, int]:
            so = so**2*np.ones(self.nobs)
        elif len(so) != self.nobs:
            raise ValueError('Provided observing system errors are not of'
                             ' length nobs.')
        else:
            print('Unknown condition for observing system errors.')
        self.so = so.reshape((-1, 1))

        # Set the boundary condition
        if BC is None:
            BC = self.ctrue
        if (type(BC) not in [float, int]):
            if  len(BC) != self.nobs:
                raise ValueError('The provided boundary condition is not '
                                 'of length 1 or nobs.')
        self.c = BC*np.ones((self.nobs, 1))

        # Solve the inversion.
        self.solve_inversion()


    def solve_inversion(self):
        # Solve the inversion
        kso_inv = self.k/self.so
        sa_inv = np.diag(1/self.sa.reshape(-1,))
        # print(self.k @ self.xa)
        # print(self.c)
        # print(self.y)

        self.shat = np.linalg.inv(sa_inv + kso_inv.T @ self.k)
        self.g = self.shat @ kso_inv.T
        self.a = np.identity(len(self.xa)) - self.shat @ sa_inv
        self.xhat = (self.xa + self.g @ (self.y - self.k @ self.xa - self.c))
        # gsum = g.sum(axis=1)

        # bc_contrib = (g @ c)
        # xa_contrib = (a @ xa)
        # tot_correct = bc_contrib + xa_contrib
        # zeta = bc_contrib/xhat#/tot_correct

        # return xhat, np.diag(a), zeta, gsum
    

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