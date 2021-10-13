import numpy as np
from numpy.linalg import inv
# from scipy.linalg import inv

import copy
import sys

sys.path.append('.')
import forward_model as fm

## -------------------------------------------------------------------------##
# Define inversion functions
## -------------------------------------------------------------------------##
def build_jacobian(x_a, y_init, BC, ts, U, L, obs_t,
                   optimize_BC=False):
    F = lambda x : fm.forward_model(x=x, y_init=y_init, BC=BC, ts=ts,
                                    U=U, L=L, obs_t=obs_t).flatten()

    # Calculate prior observations
    y_a = F(x_a)

    # Initialize the Jacobian
    K = np.zeros((len(y_a), len(x_a)))

    # Iterate through the state vector elements
    for i in range(len(x_a)):
        # Apply the perturbation to the ith state vector element
        x = copy.deepcopy(x_a)
        x[i] *= 1.5

        # Run the forward model
        y_pert = F(x)

        # Save out the result
        K[:, i] = (y_pert - y_a)/0.5

    if optimize_BC:
        # Add a column for the optimization of the boundary condition
        y_pert = fm.forward_model(x=x_a, y_init=y_init, BC=1.5*BC, ts=ts,
                                  U=U, L=L, obs_t=obs_t).flatten()
        dy_dx = ((y_pert - y_a)/0.5).reshape(-1, 1)
        K = np.append(K, dy_dx, axis=1)

    return K

def solve_inversion(x_a, s_a_vec, y, y_a, s_o_vec, k,
                    optimize_BC=False, verbose=False):
    # Get the state vector dimension and create the relative prior
    nstate = k.shape[1]
    x_a = np.ones(nstate)

    # Get the s_a_vec
    if optimize_BC:
        s_a_vec = np.append(s_a_vec, 0.075)
        print(s_a_vec)

    # Get the inverse of s_a and s_o
    s_a_inv = np.diag(1/s_a_vec)
    s_o_inv = np.diag(1/s_o_vec)

    # Calculate c
    c = y_a - k @ x_a

    # Solve the inversion
    s_hat = inv(s_a_inv + k.T @ s_o_inv @ k)
    g = s_hat @ k.T @ s_o_inv
    a = np.identity(nstate) - s_hat @ s_a_inv
    x_hat = (x_a + g @ (y - k @ x_a - c))

    if optimize_BC:
        print(x_hat)
        x_hat = x_hat[:-1]
        s_hat = s_hat[:-1, :-1]
        a = a[:-1, :-1]
        g = g[:-1, :-1]

    return x_hat, s_hat, a, g
