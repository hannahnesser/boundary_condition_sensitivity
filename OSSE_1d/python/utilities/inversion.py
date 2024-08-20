import numpy as np
import math
from copy import deepcopy as dc
import settings as s
from utilities import gcpy as gc

class OSSE:
    def __init__(self, nstate, nobs_per_cell, Cmax, L, U, init_t, total_t,
                 BCt, xt_abs, rs):
        '''
        Define an inversion object for an OSSE:
            nstate              ...
            nobs_per_cell       ...
            C 
            L
            U
            init_t
            total_t
            BCt
            xt_abs             True emissions in ppb/day. We assume 
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
        # Define dimensions of the state and observation vectors
        self.nstate = nstate
        self.nobs_per_cell = nobs_per_cell
        self.nobs = self.nstate*self.nobs_per_cell

        # Define a plotting vector 
        self.xp = np.arange(1, nstate + 1)

        # Define the grid cell length and wind speed
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
        self.BCt = BCt
        self.xt_abs = xt_abs*np.ones(nstate)

        # Initial conditions given by steady state with the true BC
        # (We don't need to use the forward model for these because 
        # every model simulation has a spin-up that will reach steady
        # state.)
        self.y0 = self.BCt + np.cumsum(self.xt_abs/self.j)

        # Pseudo observations
        self.y = (self.forward_model(x=self.xt_abs, 
                                     BC=self.BCt*np.ones(len(self.t)))
                  + np.random.RandomState(rs).normal(
                        0, 8, (self.nstate, self.nobs_per_cell))).T.flatten()

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
        # # Get times
        # if times is None:
        #     times = self.t[self.t >= self.obs_t.min()]

        # Create an empty array (grid box x time) for all
        # model output
        ys = np.zeros((self.nstate, len(self.t)))
        ys[:, 0] = self.y0

        # Iterate through the time steps
        for i, t in enumerate(self.t[1:]):
            # Do advection and emissions using the boundary condition
            # from the previous time step (since Lax Wendroff relies
            # on the concentrations across the entire domain, including
            # the boundary condition, at the previous time step) and 
            # the Courant number from the current time step. 
            # TO DO: there is some confusion about which Courant number
            # I should use.
            ynew = self.do_advection(ys[:, i], BC[i], self.C[i + 1])
            ys[:, i+1] = self.do_emissions(ynew, x)

        # Subset all output for observational times
        t_idx = gc.nearest_loc(self.obs_t, self.t)
        ys = ys[:, t_idx]

        return ys

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
                 BCt=s.BCt, xt_abs=s.xt_abs,
                 xa_abs=None, sa=s.sa, sa_BC=s.sa_BC, so=s.so, 
                 gamma=None, k=None, BC=None,
                 opt_BC=False, opt_BC_n=1, sequential=False, rs=s.random_state):
        # Inherit from the parent class
        OSSE.__init__(self, nstate, nobs_per_cell, Cmax, L, U, init_t,
                      total_t, BCt, xt_abs, rs)
        
        # Prior
        if xa_abs is None:
            self.xa_abs = np.abs(
                np.random.RandomState(rs).normal(
                    loc=25, scale=5, size=(self.nstate,)))
        else:
            self.xa_abs = xa_abs

        # Relative prior
        self.xa = np.ones(self.nstate)

        # Prior errors (ppb/day)
        if type(sa) in [float, int]:
            self.sa = (sa**2)*np.ones(self.nstate)
        else:
            self.sa = sa

        # Observational errors (ppb)
        if type(so) in [float, int]:
            self.so = (so**2)*np.ones(self.nobs)
        else:
            self.so = so
        self.gamma = gamma

        # Define the inversion boundary condition (of length of self.t)
        if BC is None:
            BC = BCt
        self.BC = BC*np.ones(len(self.t))

        # Boolean for whether we optimize the BC, and expand the previously
        # defined quantities to include BC elements
        self.opt_BC = opt_BC
        if self.opt_BC:
            self.opt_BC_n = opt_BC_n
            self._expand_inversion_for_BC(sa_BC)
        else:
            self._nstate = self.nstate

        # Boolean for sequential solution (i.e., optimize BC only, then fluxes 
        # only)
        self.sequential = sequential

        # Prior model simulation
        self.ya = self.forward_model(x=self.xa_abs, BC=self.BC).T.flatten()
        # print(self.ya)
        # self.ya = self.ya.flatten()
        # print(self.ya)

        # Build the Jacobian
        if k is None:
            self.build_jacobian()

        # Calculate c
        self.c = self.ya - self.k @ self.xa

        # Solve the inversion and get the influence length scale
        self.solve_inversion()
        self.calculate_BC_bias_metrics()
        self.remove_BC_elements()

    def _expand_inversion_for_BC(self, sa_BC):
        # Define dummy variable for longer nstate
        self._nstate = self.nstate + self.opt_BC_n

        # Expand relative prior to include opt_BC_n BC elements.
        BC_chunks = math.ceil(len(self.BC)/self.opt_BC_n)
        xa_BC = np.nanmean(
            np.pad(
                self.BC, 
                (0, (BC_chunks - self.BC.size % BC_chunks) % BC_chunks),
                mode='constant', 
                constant_values=np.NaN).reshape(-1, BC_chunks),
            axis=1)
        self.xa = np.append(self.xa, xa_BC)

        # Expand prior errors
        self.sa = np.append(self.sa, sa_BC**2*np.ones(self.opt_BC_n))

    def build_jacobian(self):
        F = lambda x : self.forward_model(x=x, BC=self.BC).T.flatten()

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
            if (self.opt_BC_n > 1):
                BC_chunks = math.ceil(len(self.BC)/self.opt_BC_n)
                ypert = []
                i = 0
                while i < len(self.BC):
                    BCpert = self.BC.copy()
                    BCpert[i:(i + BC_chunks)] += 10
                    ypert.append(
                        self.forward_model(
                            x=self.xa_abs, 
                            BC=BCpert).T.flatten().reshape(-1, 1))
                    i += BC_chunks
                ypert = np.concatenate(ypert, axis=1)
            else:
                ypert = self.forward_model(
                        x=self.xa_abs, BC=self.BC + 10).T.flatten().reshape(-1, 1)
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
        # Solve the inversion
        if self.sequential: 
            # Solve for BC only
            self._solve_inversion_equations(
                self.y, self.k[:, -self.opt_BC_n:], self.xa[-self.opt_BC_n:],
                self.c, self.sa[-self.opt_BC_n:], self.so)
            print('Sequential BC optimized : ', self.xhat)
            
            # Update parameters with output of first inversion
            ## HON: This currently only works for opt_BC_n == 1
            self.BC = self.xhat*np.ones(len(self.BC))
            # self.ya = self.forward_model(x=self.xa_abs, BC=self.BC).T.flatten()
            self.xa[-self.opt_BC_n:] = self.xhat
            self.c = self.xhat*np.ones(self.nobs)

            # Solve the inversion with the new paramaters
            self._solve_inversion_equations(
                self.y, self.k[:, :-self.opt_BC_n], self.xa[:-self.opt_BC_n],
                self.c, self.sa[:-self.opt_BC_n], self.so)
            
            # Set self.opt_BC to False because now xhat is dimension nstate
            self.opt_BC = False

        else:
            self._solve_inversion_equations(
                self.y, self.k, self.xa, self.c, self.sa, self.so)
    
    def _solve_inversion_equations(self, y, k, xa, c, sa, so):
        # Get the inverse of sa and so
        if len(sa.shape) == 1:
            sa_inv = np.diag(1/sa)
        else:
            sa_inv = np.linalg.inv(sa)
        
        if len(so.shape) == 1:
            so_inv = np.diag(1/so)
        else:
            so_inv = np.linalg.inv(so)

        # Solve for the inversion
        self.shat = np.linalg.inv(sa_inv + k.T @ so_inv @ k)
        self.g = self.shat @ k.T @ so_inv
        self.a = np.identity(len(xa)) - self.shat @ sa_inv
        self.xhat = (xa + self.g @ (y - k @ xa - c))
        self.yhat = k @ self.xhat + c

    def get_gamma(self, tol=1e-1):
        print('Finding gamma...')
        gamma = 10
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

    def calculate_BC_bias_metrics(self):
        # First, calculate the contributions from the boundary condition
        # and from the emissions to the model corrrection term, accounting
        # for differences resulting from whether or not the boundary
        # condition is optimized by the inversion.
        if self.opt_BC:
            opt_BC_n = int(self.opt_BC_n)
            self.bc_contrib = self.a[:, -opt_BC_n:] @ self.xa[-opt_BC_n:]
            self.xa_contrib = self.a[:, :-opt_BC_n] @ self.xa[:-opt_BC_n]

        else:
            self.bc_contrib = self.g @ self.c
            self.xa_contrib = self.a @ self.xa[:self.nstate]

        # # Calculate the total model correrction
        # self.tot_correct = self.g @ self.y - (self.bc_contrib + self.xa_contrib)

    def remove_BC_elements(self):
        if self.opt_BC:
            opt_BC_n = int(self.opt_BC_n)

            self.xhat_BC = self.xhat[-opt_BC_n:]
            self.xhat = self.xhat[:-opt_BC_n]
            
            self.shat = self.shat[:-opt_BC_n, :-opt_BC_n]

            self.a_BC = self.a[-opt_BC_n:, -opt_BC_n:]
            self.a = self.a[:-opt_BC_n, :-opt_BC_n]

            self.g_BC = self.g[-opt_BC_n:, :]
            self.g = self.g[:-opt_BC_n, :]
            
            self.bc_contrib = self.bc_contrib[:-opt_BC_n]
            self.xa_contrib = self.xa_contrib[:-opt_BC_n]

    def estimate_D(self, sa_bc, R):
        k = np.abs(self.U).mean()/self.L
        xD = np.append(0, np.cumsum(self.xa_abs))[:-1]
        numer = (k*sa_bc*(self.sa**0.5*self.xa_abs).mean()
                 - R*(self.sa**0.5*self.xa_abs).mean()**2 
                 - R*k**2*(self.so**0.5).mean()**2)
        denom = R*(self.sa**0.5*xD).mean()**2
        return np.sqrt(-numer/denom)*self.L
    
    # def estimate_delta_xhat_full(self, sa_bc):
    #     D = np.arange(self.L/2, self.L*self.nstate + self.L/2, self.L)
    #     xD = np.cumsum(self.xa_abs)
    #     sa_xD
    #     numer = (self.L**2*np.abs(self.U).mean()**3*sa_bc
    #              *(self.sa**0.5).mean()**2
    #              *(self.so**0.5).mean()**2
    #              *self.xa_abs)
    #     denom = (D**3*(self.sa**0.5*xD).mean()**2
    #              *(self.L**2*(self.sa**0.5*self.xa_abs).mean()**2 
    #                + np.abs(self.U).mean()**2*(self.so**0.5).mean()**2)
    #              + D**2*self.L*np.abs(self.U).mean()**2)

    def estimate_delta_xhat(self, sa_bc):
        # k = np.abs(self.U).mean()/self.L
        D = np.arange(self.L/2, self.L*self.nstate + self.L/2, self.L)
        xD = np.append(0, np.cumsum(self.xa_abs))[:-1]
        numer = self.L*np.abs(self.U).mean()*sa_bc*self.sa*self.xa_abs
        denom = (self.sa*(D**2*xD**2 + self.L**2*self.xa_abs**2) + 
                 np.abs(self.U).mean()**2*(self.so**0.5).mean()**2)
        return -numer/denom
