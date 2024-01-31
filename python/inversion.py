import numpy as np
import math
from copy import deepcopy as dc
import sys
sys.path.append('.')
import settings as s
import gcpy as gc

class OSSE:
    def __init__(self, nstate, nobs_per_cell, Cmax, L, U, init_t, total_t,
                 BC_t, x_abs_t, random_state=872):
        '''
        Define an inversion object for an OSSE:
            nstate              ...
            nobs_per_cell       ...
            C 
            L
            U
            init_t
            total_t
            BC_t
            x_abs_t             True emissions in ppb/day. We assume 
                                that the true emissions are constant across
                                the domain, so this is a scalar. Default 
                                value of 100 ppb/day.
            sa                  Relative errors for the inversion. We assume
                                that errors are constant across the domain. 
                                Default value of 0.5.
            opt_BC              A Boolean corresponding to whether the 
                                inversion optimizes the boundary condition 
                                (BC) or not. Default False.
        '''
        # Initialize the random state
        self.rs = np.random.RandomState(random_state)

        # Define dimensions of the state and observation vectors
        self.nstate = nstate
        self.nobs_per_cell = nobs_per_cell
        self.nobs = self.nstate*self.nobs_per_cell

        # Define a plotting vector 
        self.xp = np.arange(1, nstate + 1)

        # Define the Courant number, grid cell length, and wind speed
        self.L = L
        if type(U) != np.ndarray:
            U = np.array([U])
        
        # Define time step and times to sample at
        self.delta_t = np.min(Cmax*self.L/np.abs(U))
        self.t = np.arange(0, init_t + total_t + self.delta_t, self.delta_t)
        self.obs_t = np.linspace(init_t + self.delta_t, init_t + total_t, 
                                 nobs_per_cell)

        # Rescale U to be the length of the time array.
        repeat_factor = math.ceil(len(self.t)/len(U))
        self.U = np.tile(U, repeat_factor)[:len(self.t)]

        # Calculate the Courant number
        self.C = self.U*self.delta_t/self.L

        # Get rate constants
        self.j = (self.U).mean()/self.L
        self.tau = 1/self.j

        # True quantities (ppb)
        self.BC_t = BC_t
        self.x_abs_t = x_abs_t*np.ones(nstate)

        # Initial conditions given by steady state with the true BC
        self.y0 = self.BC_t + np.cumsum(self.x_abs_t/self.j)

        # Pseudo observations
        self.y = (self.y0.reshape(-1, 1) +
                  self.rs.normal(0, 0, (self.nstate, self.nobs_per_cell)))
        self.y = self.y.flatten()

class ForwardModel(OSSE):
    def forward_model(self, x, BC):
        '''
        A function that calculates the mass in each reservoir
        after a given time given the following:
            x         :    vector of emissions (ppb/s)
            y0        :    initial atmospheric condition
            BC        :    boundary condition
            ts        :    times at which to sample the model
            U         :    wind speed
            L         :    length scale for each grid box
            obs_t     :    times at which to sample the model
        '''
        # Create an empty array (grid box x time) for all
        # model output
        ys = np.zeros((len(self.y0), len(self.t)))
        ys[:, 0] = self.y0

        # Iterate through the time steps
        for i, t in enumerate(self.t[1:]):
            # Get boundary condition for the previous time step
            # We treat this as a concentration from the
            # previous time step (follow Lax Wendroff)
            try:
                _BC = BC[i]
            except:
                _BC = BC

            # Get Courant number for this time step
            try:
                _C = self.C[i + 1]
            except:
                _C = self.C

            # Do advection and emissions
            ynew = self.do_advection(ys[:, i], _BC, _C)
            ys[:, i+1] = self.do_emissions(ynew, x)

        # Subset all output for observational times
        t_idx = gc.nearest_loc(self.obs_t, self.t)
        ys = ys[:, t_idx]

        return ys.flatten()

    def do_emissions(self, y_prev, x):
        y_new = y_prev + x*self.delta_t
        return y_new

    def do_advection(self, y_prev, BC, C):
        '''
        Advection following the Lax-Wendroff scheme
        '''
        # Append the boundary conditions
        y_prev = np.append(BC, y_prev)

        # Calculate the next time step using Lax-Wendroff
        y_new = (y_prev[1:-1]
                 - C*(y_prev[2:] - y_prev[:-2])/2
                 + C**2*(y_prev[2:] - 2*y_prev[1:-1] + y_prev[:-2])/2)

        # Update the last grid cell using upstream
        y_new = np.append(y_new, y_prev[-1] - C*(y_prev[-1] - y_prev[-2]))

        return y_new

class Inversion(ForwardModel):
    def __init__(self, nstate=s.nstate, nobs_per_cell=s.nobs_per_cell, 
                 Cmax=s.Cmax, L=s.L, U=s.U, 
                 init_t=s.init_t, total_t=s.total_t,
                 BC_t=s.BC_t, x_abs_t=s.x_abs_t,
                 xa_abs=None, sa=s.sa, sa_BC=s.sa_BC, so=s.so, 
                 gamma=None, k=None, BC=None,
                 opt_BC=False, opt_BC_n=1):
        # Inherit from the parent class
        OSSE.__init__(self, nstate, nobs_per_cell, Cmax, L, U, init_t,
                      total_t, BC_t, x_abs_t)

        # Define the inversion boundary condition
        if BC is None:
            self.BC = self.BC_t
        else:
            self.BC = BC

        # Boolean for whether we optimize the BC
        self.opt_BC = opt_BC
        if self.opt_BC:
            self._nstate = self.nstate + opt_BC_n
        else:
            self._nstate = self.nstate

        # Prior
        if xa_abs is None:
            self.xa_abs = np.abs(self.rs.normal(loc=80, scale=40, 
                                                size=(self.nstate,)))
        else:
            self.xa_abs = xa_abs

        # Relative prior
        self.xa = np.ones(self.nstate)
        if self.opt_BC:
            if (opt_BC_n > 1):
                BC_chunks = math.ceil(len(self.BC)/opt_BC_n)
                xa_BC = np.nanmean(
                    np.pad(
                        self.BC, 
                        (0, (BC_chunks - self.BC.size % BC_chunks) % BC_chunks),
                        mode='constant', 
                        constant_values=np.NaN).reshape(-1, BC_chunks),
                    axis=1)
                self.xa = np.append(self.xa, xa_BC)
            else:
                self.xa = np.append(self.xa, np.mean(self.BC))

        # Prior errors (ppb/day)
        if type(sa) == float:
            self.sa = (sa**2)*np.ones(self.nstate)
        else:
            self.sa = sa

        if self.opt_BC:
            self.sa = np.append(self.sa, 50**2*np.ones(opt_BC_n))

        # Observational errors (ppb)
        self.so = (so**2)*np.ones(self.nobs)
        self.gamma = gamma

        # Prior model simulation
        self.ya = self.forward_model(x=self.xa_abs, BC=self.BC).flatten()

        # Build the Jacobian
        if k is None:
            self.build_jacobian()

        # Calculate c
        self.c = self.ya - self.k @ self.xa

        # Solve the inversion and get the influence length scale
        self.solve_inversion()
        self.calculate_ILS()
        self.remove_BC_elements()

    def build_jacobian(self):
        F = lambda x : self.forward_model(x=x, BC=self.BC).flatten()

        # Initialize the Jacobian
        k = np.zeros((self.nobs, self.nstate))

        # Iterate through the state vector elements
        for i in range(self.nstate):
            # Apply the perturbation to the ith state vector element
            x = dc(self.xa_abs)
            x[i] *= 1.5

            # Run the forward model
            ypert = F(x)

            # Save out the result
            k[:, i] = (ypert - self.ya)/0.5

        if self.opt_BC:
            # Add a column for the optimization of the boundary condition
            if (self._nstate - self.nstate > 1) and (len(self.BC) > 1):
                BC_chunks = math.ceil(
                    len(self.BC)/(self._nstate - self.nstate))
                ypert = []
                i = 0
                while i < len(self.BC):
                    BCpert = self.BC.copy()
                    BCpert[i:(i + BC_chunks)] += 10
                    ypert.append(
                        self.forward_model(
                            x=self.xa_abs, 
                            BC=BCpert).flatten().reshape(-1, 1))
                    i += BC_chunks
                ypert = np.concatenate(ypert, axis=1)
            else:
                ypert = self.forward_model(
                    x=self.xa_abs, BC=self.BC + 10).flatten().reshape(-1, 1)
            k = np.append(k, (ypert - self.ya.reshape(-1, 1))/10, axis=1)

        self.k = k

    def solve_inversion(self):
        # Adjust the magnitude of So using a gamma (for similarity
        # to a real inversion)
        if self.gamma is None:
            self.get_gamma()
        else:
            self.so = self.so/self.gamma
        self._solve_inversion()

    def _solve_inversion(self):
        # Get the inverse of sa and so
        sa_inv = np.diag(1/self.sa)
        so_inv = np.diag(1/self.so)

        # Solve the inversion
        self.shat = np.linalg.inv(sa_inv + self.k.T @ so_inv @ self.k)
        self.g = self.shat @ self.k.T @ so_inv
        self.a = np.identity(len(self.xa)) - self.shat @ sa_inv
        self.xhat = (self.xa + self.g @ (self.y - self.k @ self.xa - self.c))

    def get_gamma(self, tol=1e-1):
        print('Finding gamma...')
        gamma = 20
        gamma_not_found = True
        so_orig = dc(self.so)
        while gamma_not_found:
            self.so = so_orig/gamma
            self._solve_inversion()
            cost = self.cost_prior()/self.nstate
            print(f'{gamma:.4f}: {cost:.3f}')
            if np.abs(cost - 1) <= tol:
                gamma_not_found = False
            elif cost > 1:
                gamma /= 2
            elif cost < 1:
                gamma *= 1.5
        self.gamma = gamma
        print('Gamma found! Adjusting So.')
        print('-'*70)

    def cost_function(self):
        ...

    def cost_prior(self):
        return (((self.xhat - self.xa)**2)/self.sa).sum()

    def cost_obs(self):
        ...

    def calculate_ILS(self, ILS_threshold=0.5):
        # First, calculate the contributions from the boundary condition
        # and from the emissions to the model corrrection term, accounting
        # for differences resulting from whether or not the boundary
        # condition is optimized by the inversion.
        if self.opt_BC:
            opt_BC_n = int(self._nstate - self.nstate)
            self.bc_contrib = self.a[:, -opt_BC_n:] @ self.xa[-opt_BC_n:]
            self.xa_contrib = self.a[:, :-opt_BC_n] @ self.xa[:-opt_BC_n]
        else:
            self.bc_contrib = self.g @ self.c
            self.xa_contrib = self.a @ self.xa[:self.nstate]

        # Calculate the total model correrction
        self.tot_correct = self.g @ self.y - (self.bc_contrib + self.xa_contrib)

        # HN just now removed absolute values

        # Then, use the ratio of the correction attributable to the 
        # boundary condition together with the defined threshold
        # to define the influence length scale.
        try:
            self.ils = np.where((self.bc_contrib/self.tot_correct) < ILS_threshold)[0][0]
        except:
            if np.all(self.bc_contrib/self.tot_correct < ILS_threshold):
                self.ils = 0
            elif np.all(self.bc_contrib/self.tot_correct >= ILS_threshold):
                self.ils = self.nstate
            else:
                self.ils = None

    def remove_BC_elements(self):
        if self.opt_BC:
            opt_BC_n = int(self._nstate - self.nstate)

            self.xhat_BC = self.xhat[-opt_BC_n:]
            self.xhat = self.xhat[:-opt_BC_n]
            
            self.shat = self.shat[:-opt_BC_n, :-opt_BC_n]
            self.a = self.a[:-opt_BC_n, :-opt_BC_n]
            self.g = self.g[:-opt_BC_n, :]
            self.bc_contrib = self.bc_contrib[:-opt_BC_n]
            self.xa_contrib = self.xa_contrib[:-opt_BC_n]
            self.tot_correct = self.tot_correct[:-opt_BC_n]
