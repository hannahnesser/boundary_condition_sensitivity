import numpy as np
import sys
sys.path.append('.')
import gcpy as gc

## -------------------------------------------------------------------------##
# Define forward model
## -------------------------------------------------------------------------##
def forward_model(x, y_init, BC, ts, U, L, obs_t):
    '''
    A function that calculates the mass in each reservoir
    after a given time given the following:
        x         :    vector of emissions (ppb/s)
        y_init    :    initial atmospheric condition
        BC        :    boundary condition
        ts        :    times at which to sample the model
        U         :    wind speed
        L         :    length scale for each grid box
        obs_t     :    times at which to sample the model
    '''
    # Create an empty array (grid box x time) for all
    # model output
    ys = np.zeros((len(y_init), len(ts)))
    ys[:, 0] = y_init

    # Get time steps
    delta_t = np.diff(ts)

    # Iterate through the time steps
    for i, t in enumerate(ts[1:]):
        # Get boundary condition
        try:
            bc = BC[i]
        except:
            bc = BC

        # Get
        y_new = do_emissions(x, ys[:, i], delta_t[i])
        ys[:, i+1] = do_advection(x, y_new, bc, delta_t[i], U, L)

    # Subset all output for observational times
    t_idx = gc.nearest_loc(obs_t, ts)
    ys = ys[:, t_idx]

    return ys

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
