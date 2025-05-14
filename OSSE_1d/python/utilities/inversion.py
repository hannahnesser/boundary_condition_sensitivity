import numpy as np
import math
from copy import deepcopy as dc
from utilities import gcpy as gc
from utilities import utils
project_dir, config = utils.setup()

class OSSE:
    def __init__(self, nstate, nobs_per_cell, Cmax, L, U, init_t, total_t,
                 BCt, xt_abs, obs_err, buffer, rs):
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
        y_err = np.random.RandomState(rs).normal(
            0, obs_err, (self.nstate, self.nobs_per_cell))
        if buffer:
            y_err = np.vstack([y_err[-1, :], y_err[:-1, :]])
        self.y = (self.forward_model(x=self.xt_abs, 
                                     BC=self.BCt*np.ones(len(self.t)))
                  + y_err).T.flatten()

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
    def __init__(
            self, 
            nstate=config['nstate'], 
            nobs_per_cell=config['nobs_per_cell'], 
            Cmax=config['Cmax'], 
            L=config['L'], 
            U=config['U'], 
            init_t=config['init_t'],
            total_t=config['total_t'],
            BCt=config['BCt'], 
            xt_abs=config['xt_abs'], 
            obs_err=config['obs_err'],
            xa_abs=None, 
            sa=config['sa'], 
            sa_BC=config['sa_BC'], 
            so=config['so'], 
            gamma=None, 
            k=None, 
            BC=None,
            opt_BC=False, 
            opt_BC_n=1, 
            buffer=False,
            sequential=False, 
            rs=config['random_state']):
        # Inherit from the parent class
        if buffer:
            nstate += 1
        OSSE.__init__(self, nstate, nobs_per_cell, Cmax, L, U, init_t,
                      total_t, BCt, xt_abs, obs_err, buffer, rs)
        
        # Prior
        if xa_abs is None:
            self.xa_abs = np.abs(
                np.random.RandomState(rs).normal(
                    loc=25, scale=5, size=(self.nstate,)))
        else:
            self.xa_abs = xa_abs
        
        if buffer:
            self.xa_abs = np.append(self.xa_abs[-1], self.xa_abs[:-1])

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

        # Boolean for whether to use the buffer grid cell
        self.buffer = buffer
        if self.buffer:
            self.p = self.estimate_p(sa_BC)
            # print('  Buffer scale factor : ', self.p)
            self.sa[0] *= self.p**2

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
        self.remove_buffer_elements()

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
                constant_values=np.nan).reshape(-1, BC_chunks),
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
        # print(f'  Prior cost function : {self.cost_prior():.2f} (n={self.xhat.shape})')

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

            self.shat_full = self.shat
            self.shat = self.shat[:-opt_BC_n, :-opt_BC_n]

            self.a_full = self.a
            self.a_BC = self.a[-opt_BC_n:, -opt_BC_n:]
            self.a = self.a[:-opt_BC_n, :-opt_BC_n]

            self.g_BC = self.g[-opt_BC_n:, :]
            self.g = self.g[:-opt_BC_n, :]
            
            self.bc_contrib = self.bc_contrib[:-opt_BC_n]
            self.xa_contrib = self.xa_contrib[:-opt_BC_n]

    def remove_buffer_elements(self):
        if self.buffer:
            self.xhat_buffer = self.xhat[0]
            self.xhat = self.xhat[1:]

            self.shat_full = self.shat
            self.shat = self.shat[1:, 1:]

            self.sa_full = self.sa
            self.sa = self.sa[1:]

            self.a_full = self.a
            self.a_BC = self.a[0, 0]
            self.a = self.a[1:, 1:]

            self.g_BC = self.g[0, :]
            self.g = self.g[1:, :]
            
            # self.bc_contrib = self.bc_contrib[1:]
            # self.xa_contrib = self.xa_contrib[1:]

    def estimate_D(self, sa_bc, R):
        # k = np.abs(self.U).mean()/self.L
        # xa = self.xa_abs
        # xd = np.append(0, np.cumsum(self.xa_abs))[:-1]
        # so = (self.so**0.5).mean()**2
        # D = np.abs(xa*k*sa_bc/(R*xd**2) - k**2*so/(self.sa*xd**2) - xa**2/xd**2)
        # return self.L*np.sqrt(D)
        delta_xhat = np.abs(self.estimate_delta_xhat(sa_bc))
        return np.where(delta_xhat < R*delta_xhat.max())

    # def preview_1d(self, sa_bc):
    #     U = np.abs(self.U).mean()
    #     k = self.L/U*np.tril(np.ones((self.nstate, self.nstate)))*self.xa_abs.mean()
    #     so = ((self.so.mean()/self.nobs_per_cell)**0.5)**2 # individual 
    #     so_inv = np.diag(1/(so*np.ones(self.nstate)))
    #     g = np.linalg.inv(np.diag(1/self.sa) + k.T @ so_inv @ k) @ k.T @ so_inv
    #     return -sa_bc*g.sum(axis=1)/self.xa

    # This is using the full 2x2 example, but it generates an estimate that is
    # too low
    def preview_2d(self, sa_bc):
        # Transport
        D = np.arange(self.L, self.L*self.nstate + self.L, self.L)
        j = D/self.L
        U = self.U.mean()
        k_i = self.L/U
        k_up = (D/U)[:-1]

        # Prior and prior uncertainty
        sa_i = (self.xa_abs**2*self.sa)[1:]
        sa_up = (np.cumsum(self.xa_abs)**2*self.sa)[:-1]
        # print(self.xa_abs)
        # print(self.xa_abs[1:])
        # print(np.cumsum(self.xa_abs))

        # Observing system errors 
        so_i = ((self.so.mean()/self.nobs_per_cell)**0.5)**2 # individual 
        so_up = (so_i/j)[:-1]

        # Secondary quantitites
        R_i = so_i/(k_i**2*sa_i)
        R_up = so_up/(k_up**2*sa_up)
        beta = j[1:]**2*sa_up/sa_i + 1
        
        # Predict error
        delta_xhat_0 = (sa_bc/k_i)/(1 + R_i[0])
        delta_xhat = (sa_bc/k_i)*R_up/(R_up*R_i + R_i + beta*R_up + 1)

        # While we're at it, we'll estimate an influence length scale using
        # the 1D approximation
        so_mean = (np.mean(so_i**0.5)**2)
        sa = np.mean(self.sa**0.5)**2*np.mean(self.xa_abs)**2

        # print('Estimating R')
        # print(f'  Mean K : {np.mean(k_i):.2f}')
        # print(f'  Mean sa : {sa**0.5:.2f}')
        # print(f'  Mean so : {so_mean**0.5}')
        # print(f'  R : {so_mean/(np.mean(k_i)**2*sa):.2f}')
        R = so_mean/(np.mean(k_i)**2*sa)
        print('  R : ', 1/R)

        return - np.append(delta_xhat_0, delta_xhat)/self.xa_abs, R
        
        # kx_i = k_i * self.xa_abs
        # kx_up = k_up * x_up
        # sa_so_up = self.sa / so_up
        # so_i_so_up = so / so_up
        # so_i_sa = so / self.sa

        # num = (sa_bc * kx_i)
        # den = kx_up**2 * ( sa_so_up * kx_i**2 + so_i_so_up + 1 ) + kx_i**2 + so_i_sa
        # delta_xhat = - num / den

        # return delta_xhat

    def estimate_p(self, sa_bc):
        try: 
            Umin = self.U.min()
            Umax = self.U.max()
        except:
            Umin = self.U
            Umax = self.U
        tau_min = self.L/Umax
        tau_max = self.L/Umin
        sa_min = (self.sa**0.5*self.xa_abs).min()**2
        sa_max = (self.sa**0.5*self.xa_abs).max()**2
        R_min = tau_min**2*sa_min/self.so.max()*self.nobs_per_cell
        R_max = tau_max**2*sa_max/self.so.max()*self.nobs_per_cell
        R_sqrt_min = np.minimum(np.sqrt(R_min + 2), np.sqrt(R_max + 2))
        R_sqrt_max = np.maximum(np.sqrt(R_min + 2), np.sqrt(R_max + 2))
        p_min = sa_bc/(tau_max*sa_max*R_sqrt_max)
        p_max = sa_bc/(tau_min*sa_min*R_sqrt_min)
        print('  Buffer scale factor : ', p_min, p_max)
        return p_max
        # rat_min = (sa_bc/((self.L/Umin)*self.sa.mean()**0.5*self.xa_abs.max()))**2
        # rat_max = (sa_bc/((self.L/Umax)*self.sa.mean()**0.5*self.xa_abs.min()))**2
        # return (rat_min + 1)**0.5, (rat_max + 1)**0.5

    # def estimate_delta_xhat(self, sa_bc):
    #     D = np.arange(self.L/2, self.L*self.nstate + self.L/2, self.L)
    #     xD = np.append(0, np.cumsum(self.xa_abs))[:-1] + self.xa_abs/2
    #     U = np.abs(self.U).mean()
    #     so = ((self.so.mean()/self.nobs_per_cell)**0.5)**2

    #     kL = self.L/U
    #     kD = D/U

    #     numer = kL*self.sa*self.xa_abs*sa_bc # Not xa_abs^2 because we want relative change
    #     denom = (self.sa*(kD**2*xD**2 + kL**2*self.xa_abs**2) + so)
    #     # m2 ppb2/hr2
        
    #     # # Calculate covariance
    #     # cov_xD = self.L**2*self.sa**2*self.xa_abs**2 + U**2*self.sa*so
    #     # # m2 ppb2/hr2
    #     # cov_x = D**2*self.sa**2*xD**2 + U**2*self.sa*so # m2 ppb2/hr2
    #     # cov_xxD = -D*self.L*self.sa**2*self.xa_abs*xD

    #     return -numer/denom #, cov_xD/denom, cov_x/denom, cov_xxD/denom

    # def estimate_avker(self):
    #     so = ((self.so.mean()/self.nobs_per_cell)**0.5)**2
    #     U = np.abs(self.U).mean()
    #     kL = self.L/U
    #     a_est = self.sa / (self.sa + so/kL**2)
    #     return a_est

    # def estimate_a():
    #     k_est = np.ones()