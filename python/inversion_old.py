import numpy as np
from numpy.linalg import inv as npinv
# from scipy.linalg import inv

import copy
import sys

sys.path.append('.')
import forward_model as fm

## -------------------------------------------------------------------------##
# Define inversion functions
## -------------------------------------------------------------------------##
def solve_inversion(xa, sa_vec, y, so_vec, k, ya, opt_BC=False):
    # Get the inverse of sa and so
    sa_inv = np.diag(1/sa_vec)
    so_inv = np.diag(1/so_vec)

    # Calculate c
    c = ya - k @ xa

    # Solve the inversion
    shat = npinv(sa_inv + k.T @ so_inv @ k)
    g = shat @ k.T @ so_inv
    a = np.identity(len(xa)) - shat @ sa_inv
    xhat = (xa + g @ (y - k @ xa - c))

    if opt_BC:
        xhat = xhat[:-1]
        shat = shat[:-1, :-1]
        a = a[:-1, :-1]
        g = g[:-1, :-1]

    return xhat, shat, a, g

def get_gain_matrix(sa_vec, so_vec, k, opt_BC=False):
    # Get the sa_vec
    if opt_BC:
        sa_vec = np.append(sa_vec, 0.075)

    # Get the inverse of s_a and s_o
    sa_inv = np.diag(1/sa_vec)
    so_inv = np.diag(1/so_vec)

    # Solve the inversion
    shat = npinv(sa_inv + k.T @ so_inv @ k)
    g = shat @ k.T @ so_inv

    return np.array(g)

def band_width(g):
    g_max = g.max(axis=1).reshape(-1, 1)*1e-3
    band_width = (np.abs(g) > g_max).sum(axis=1).max()
    band_width /= g.shape[1]
    return band_width

def e_folding_length(g):
    gsum = np.abs(g.sum(axis=1))
    tau = -np.arange(1, len(gsum))/np.log(gsum[1:]/gsum[0])
    # print(tau)
    # tau = -(np.arange(2, len(gsum))/np.log(gsum[1:]/gsum[0]))
    return tau.max()

def influence_length(g, k, xa, c):
    xa_contrib = g @ k @ xa
    bc_contrib = g @ c
    bc_contrib_rel = bc_contrib/(xa_contrib + bc_contrib)
    ils = np.where(bc_contrib_rel < 0.5)[0][0]
    return ils

def xhat_err(xhat, xhat_true, ils):
    xhat_diff = np.abs(xhat - xhat_true).reshape(-1,)
    in_ils_err = xhat_diff[:ils].mean()
    out_ils_err = xhat_diff[ils:].mean()
    return in_ils_err, out_ils_err

def oscillating_bc_pert(t, y, amp, freq, phase):
    return y + amp*np.sin(freq*t + phase)

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

## -------------------------------------------------------------------------##
## Generate variables for the true inversion solved with the true BC
## and standard conditions
## -------------------------------------------------------------------------##


