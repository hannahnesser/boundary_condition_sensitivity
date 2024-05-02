import numpy as np
import math
from copy import deepcopy as dc
import settings as s

class OSSE:
    def __init__(self, nstate, nobs_per_cell, Cmax, L, U, init_t, total_t,
                 BCt, xt_abs, rs):
        '''
        Define an object for an OSSE. Default values for each of these 
        variables is set and described in the Inversion object (below).
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
    '''
    The ForwardModel object inherits from the OSSE. The forward model uses
    a Lax-Wendroff scheme for the first nstate - 1 grid cells and switches
    to an upstream method for the last grid cell (to avoid the need to 
    define the downwind boundary condition). 
    '''
    def forward_model(self, x, BC):
        '''
        A function that calculates the mass in each reservoir
        for each time step contained in self.t for a given:
          x    [array of length nstate ] Vector of emissions (ppb/s)
          BC   [float or array] Boundary condition [ppb]
        '''
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
            # the Courant number from the current time step. [Check]
            ynew = self.do_advection(ys[:, i], BC[i], self.C[i + 1])
            ys[:, i+1] = self.do_emissions(ynew, x)

        # Subset all output for observational times
        t_idx = ForwardModel.nearest_loc(self.obs_t, self.t)
        ys = ys[:, t_idx]

        return ys

    def do_emissions(self, y_prev, x):
        y_new = y_prev + x*self.delta_t
        return y_new

    def do_advection(self, y_prev, BC, C):
        # Append the boundary conditions
        y_prev = np.append(BC, y_prev)

        # Calculate the next time step using Lax-Wendroff
        y_new = (y_prev[1:-1]
                 - C*(y_prev[2:] - y_prev[:-2])/2
                 + C**2*(y_prev[2:] - 2*y_prev[1:-1] + y_prev[:-2])/2)

        # Update the last grid cell using upstream
        y_new = np.append(y_new, y_prev[-1] - C*(y_prev[-1] - y_prev[-2]))

        return y_new
    
    @staticmethod
    def nearest_loc(data, compare_data):
        indices = np.abs(compare_data.reshape(-1, 1) -
                        data.reshape(1, -1)).argmin(axis=0)
        return indices

class Inversion(ForwardModel):
    '''
    The Inversion object inherits from ForwardModel, which inherits from
    OSSE. All default arguments are set here for the sake of convenience 
    (e.g., so that we can call test = inv.Inversion() and control the defaults
    in OSSE). The parameter defaults are set in settings.py unless otherwise
    mentioned and include:
      nstate          [int] The number of gridded state vector elements,
                      excluding any boundary condition elements.
      nobs_per_cell   [int] The number of observations per grid cell for the
                      OSSE to generate.
      Cmax            [float] The maximum Courant number, which is used to set
                      the time step.
      L               [float] The length of a single grid cell in km.
      U               [float or array] The wind speed in km/day. If an array,
                      it will be expanded to fill the entire simulation period
                      by repeating the array.
      init_t          [float] The start time for the inversion (excluding
                      spin-up) in hours.
      total_t         [float] The total duration of the simulation (including
                      spin-up) in hours.
      BCt             [float] The true boundary condition in ppb. This is used
                      to generate the pseudo-observations.
      xt_abs          [float] The true emissions in ppb/day used to generate
                      the pseudo-observations. Currently, this is treated as a
                      scalar so that the true emissions are constant across the 
                      domain.
      xa_abs          [float or array] The prior emissions in ppb/day with a 
                      default of random values with mean 25 ppb/day and
                      standard deviation 5 ppb/day.
      sa              [float or array] The prior error standard deviation in 
                      relative (percent) space.
      so              [float or array] The observing system standard deviation
                      in ppb.
      k               [array of nstate x nobs] The Jacobian matrix for the 
                      inversion. The default is to construct the matrix using
                      ForwardModel.
      BC              [float or array] The boundary condition for the forward
                      model/inversion. This is separate from BCt, which is used
                      to generate pseudo-observations. The default is BCt.
      rs              [int] An integer corresponding to the random state, to
                      allow exploration of the sensitivity of the result to the
                      prior emissions.

    '''
    def __init__(
            self, nstate=s.nstate, nobs_per_cell=s.nobs_per_cell, 
            Cmax=s.Cmax, L=s.L, U=s.U, init_t=s.init_t, total_t=s.total_t,
            BCt=s.BCt, xt_abs=s.xt_abs, xa_abs=None, sa=s.sa, so=s.so, 
            k=None, BC=None, rs=s.random_state):
        
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

        # Define the inversion boundary condition (of length of self.t)
        if BC is None:
            BC = BCt
        self.BC = BC*np.ones(len(self.t))

        # Prior model simulation
        self.ya = self.forward_model(x=self.xa_abs, BC=self.BC).T.flatten()

        # Build the Jacobian
        if k is None:
            self.build_jacobian()

        # Calculate c
        self.c = self.ya - self.k @ self.xa

        # Solve the inversion
        self.solve_inversion()

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

        self.k = k

    def solve_inversion(self):
        # Get the inverse of the error covariance matrices (currently this
        # assumes both matrices are diagonal)
        sa_inv = np.diag(1/self.sa)
        so_inv = np.diag(1/self.so)

        # Solve for the inversion
        self.shat = np.linalg.inv(sa_inv + self.k.T @ so_inv @ self.k)
        self.g = self.shat @ self.k.T @ so_inv
        self.a = np.identity(self.nstate) - self.shat @ sa_inv
        self.xhat = (self.xa + self.g @ (self.y - self.k @ self.xa - self.c))
        self.yhat = self.k @ self.xhat + self.c

    @staticmethod
    def rmse(diff):
        return np.sqrt(np.mean(diff**2))

    @staticmethod
    def add_quad(data):
        return np.sqrt((data**2).sum())

    @staticmethod
    def rel_err(data, truth):
        return (data - truth)/truth

