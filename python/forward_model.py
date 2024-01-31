import numpy as np
import copy
import sys
sys.path.append('.')
import gcpy as gc

## -------------------------------------------------------------------------##
# Define forward model
## -------------------------------------------------------------------------##
def forward_model(x, y0, BC, ts, U, L, obs_t):
    '''
    A function that calculates the mass in each reservoir
    after a given time given the following:
        x         :    vector of emissions (ppb/s)
        y0    :    initial atmospheric condition
        BC        :    boundary condition
        ts        :    times at which to sample the model
        U         :    wind speed
        L         :    length scale for each grid box
        obs_t     :    times at which to sample the model
    '''
    # Create an empty array (grid box x time) for all
    # model output
    ys = np.zeros((len(y0), len(ts)))
    ys[:, 0] = y0

    # Get time steps
    delta_t = np.diff(ts)

    # Iterate through the time steps
    for i, t in enumerate(ts[1:]):
        # Get boundary condition
        try:
            bc = BC[i]
        except:
            bc = BC

        # Do advection and emissions
        ynew = do_advection(x, ys[:, i], bc, delta_t[i], U, L)
        ys[:, i+1] = do_emissions(x, ynew, delta_t[i])

    # Subset all output for observational times
    t_idx = gc.nearest_loc(obs_t, ts)
    ys = ys[:, t_idx]

    return ys.flatten()

def do_emissions(x, y_prev, delta_t):
    y_new = y_prev + x*delta_t
    return y_new

def do_advection(x, y_prev, BC, delta_t, U, L):
    '''
    Advection following the Lax-Wendroff scheme
    '''
    # Calculate the courant number
    C = U*delta_t/L

    # Append the boundary conditions
    y_prev = np.append(BC, y_prev)

    # Calculate the next time step using Lax-Wendroff
    y_new = (y_prev[1:-1]
             - C*(y_prev[2:] - y_prev[:-2])/2
             + C**2*(y_prev[2:] - 2*y_prev[1:-1] + y_prev[:-2])/2)

    # Update the last grid cell using upstream
    y_new = np.append(y_new, y_prev[-1] - C*(y_prev[-1] - y_prev[-2]))

    return y_new

def build_jacobian(xa, y0, BC, ts, U, L, obs_t, opt_BC=False):
    F = lambda x : forward_model(x=x, y0=y0, BC=BC, ts=ts,
                                  U=U, L=L, obs_t=obs_t).flatten()

    # Calculate prior observations
    ya = F(xa)

    # Initialize the Jacobian
    K = np.zeros((len(ya), len(xa)))

    # Iterate through the state vector elements
    for i in range(len(xa)):
        # Apply the perturbation to the ith state vector element
        x = copy.deepcopy(xa)
        x[i] *= 1.5

        # Run the forward model
        ypert = F(x)

        # Save out the result
        K[:, i] = (ypert - ya)/0.5

    if opt_BC:
        # Add a column for the optimization of the boundary condition
        ypert = forward_model(x=xa, y0=y0, BC=1.5*BC, ts=ts,
                              U=U, L=L, obs_t=obs_t).flatten()
        dy_dx = ((ypert - ya)/0.5).reshape(-1, 1)
        K = np.append(K, dy_dx, axis=1)

    return K