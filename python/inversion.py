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
        x_hat = x_hat[:-1]
        s_hat = s_hat[:-1, :-1]
        a = a[:-1, :-1]
        g = g[:-1, :-1]

    return x_hat, s_hat, a, g

def get_gain_matrix(s_a_vec, s_o_vec, k, optimize_BC=False):
    # Get the s_a_vec
    if optimize_BC:
        s_a_vec = np.append(s_a_vec, 0.075)

    # Get the inverse of s_a and s_o
    s_a_inv = np.diag(1/s_a_vec)
    s_o_inv = np.diag(1/s_o_vec)

    # Solve the inversion
    s_hat = inv(s_a_inv + k.T @ s_o_inv @ k)
    g = s_hat @ k.T @ s_o_inv

    return np.array(g)

def band_width(g):
    g_max = g.max(axis=1).reshape(-1, 1)*1e-3
    band_width = (np.abs(g) > g_max).sum(axis=1).max()
    return band_width

def influence_length(xhat, xhat_true, threshold=0.1, relative=True):
    # if relative:
    #     print(f'Applying a threshold of {threshold:.2f} (relative).')
    # else:
    #     print(f'Applying a threshold of {threshold:.2f} (absolute).')

    xhat_diff = np.abs(xhat - xhat_true).reshape(-1,)
    print(xhat_diff)
    print(np.where(xhat_diff < threshold))
    ils = np.where(xhat_diff < threshold)[0][0]
    return ils

def xhat_err(xhat, xhat_true, ils):
    xhat_diff = np.abs(xhat - xhat_true).reshape(-1,)
    in_ils_err = xhat_diff[:ils].mean()
    out_ils_err = xhat_diff[ils:].mean()
    return in_ils_err, out_ils_err


    # # calculate influence length scale
    # g_sum = g.sum(axis=1).reshape(-1, 1)*bc_err
    # ils = np.where(np.abs(g_sum) > 10)[0]
    # ils_comp = np.arange(0, len(ils))
    # ils = (ils == ils_comp).sum()
    # # g_sum.argwhere(g_sum > 10)[0]

# def avg_err(xhat, xhat_true):


## -------------------------------------------------------------------------##
## Cluster functions
## -------------------------------------------------------------------------##
def match_data_to_clusters(data, clusters, default_value=0):
    '''
    Matches inversion data to a cluster file.
    Parameters:
        data (np.array)        : Inversion data. Must have the same length
                                 as the number of clusters, and must be
                                 sorted in ascending order of cluster number
                                 - i.e. [datapoint for cluster 1, datapoint for
                                 cluster 2, datapoint for cluster 3...]
        clusters (xr.Datarray) : 2d array of cluster values for each gridcell.
                                 You can get this directly from a cluster file
                                 used in an analytical inversion.
                                 Dimensions: ('lat','lon')
        default_value (numeric): The fill value for the array returned.
    Returns:
        result (xr.Datarray)   : A 2d array on the GEOS-Chem grid, with
                                 inversion data assigned to each gridcell based
                                 on the cluster file.
                                 Missing data default to the default_value.
                                 Dimensions: same as clusters ('lat','lon').
    '''
    # check that length of data is the same as number of clusters
    clust_list = np.unique(clusters)[np.unique(clusters)!=0] # unique, nonzero clusters
    assert len(data)==len(clust_list), (f'Data length ({len(data)}) is not the same as '
                                        f'the number of clusters ({len(clust_list)}).')

    # build a lookup table from data.
    #    data_lookup[0] = default_value (value for cluster 0),
    #    data_lookup[1] = value for cluster 1, and so forth
    data_lookup = np.append(default_value, data)

    # use fancy indexing to map data to 2d cluster array
    cluster_index = clusters.squeeze().data.astype(int).tolist()
    result = clusters.copy().squeeze()         # has same shape/dims as clusters
    result.values = data_lookup[cluster_index] # map data to clusters

    return result

