import numpy as np
import pandas as pd
import copy
import os
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.lines import Line2D

# Custom packages
import sys
sys.path.append('.')
import gcpy as gc
import forward_model as fm
import inversion as inv
import plot
import config
config.SCALE = config.PRES_SCALE
config.BASE_WIDTH = config.PRES_WIDTH
config.BASE_HEIGHT = config.PRES_HEIGHT
import format_plots as fp

np.set_printoptions(precision=3, linewidth=300, suppress=True)

## -------------------------------------------------------------------------##
# File Locations
## -------------------------------------------------------------------------##
plot_dir = '../plots'

## -------------------------------------------------------------------------##
# Define unchanging model parameters
## -------------------------------------------------------------------------##
optimize_BC = False

# Define the parameters of our simple forward model
U = 5/3600 # windspeed (5 km/hr in km/s)
L = 25 #12.5 # grid cell length (25 km)
j = U/L # transfer coeficient (s-1)
tau = 1/j

# Dimensions of the inversion quantities
nstate = 20 #20 #30 # state vector
nobs_per_cell = 15 #15 #15 #30
nobs = nobs_per_cell*nstate # observation vector

# Define the times
init_t = 150*3600
total_t = 150*3600 # time in seconds

# Define the true emissions, including the boundary conditions
BC_t = 1900 # ppb
x_t = 100*np.ones(nstate)/(3600*24) # true (ppb/s)

# Define the initial conditions
y_init = [BC_t + x_t[0]/j]
for i in range(1, nstate):
    y_init.append(y_init[-1] + x_t[i]/j)
y_init = np.array(y_init)
y_init = copy.deepcopy(y_init)

## -------------------------------------------------------------------------##
# Generate pseudo-observations and define observational error
## -------------------------------------------------------------------------##
# Seed the random number generator
from numpy.random import RandomState
rs = RandomState(728)

# Create pseudo-observations
random_noise = rs.normal(0, 10, (nstate, nobs_per_cell))
y = y_init.reshape(-1,1) + random_noise

# Define the observational errror
s_o_vec = 15*np.ones(nobs) #15
s_o_vec **= 2

## -------------------------------------------------------------------------##
# Model parameters that are calculated from user inputs
## -------------------------------------------------------------------------##
nobs = nobs_per_cell*nstate # observation vector

# Define the times at which we sample the forward model and the observations
C = 0.5 # Courant number
delta_t = C*L/U # seconds
t = np.arange(0, init_t + total_t + delta_t, delta_t)
obs_t = np.linspace(init_t + delta_t, init_t + total_t, nobs_per_cell)

## -------------------------------------------------------------------------##
# Calculate the Jacobian
## -------------------------------------------------------------------------##
x_a = np.abs(rs.normal(loc=70, scale=40, size=(nstate,))/(3600*24)) # prior (ppb/s)
y_a = fm.forward_model(x_a, y_init, BC_t, t, U, L, obs_t)
K_t = inv.build_jacobian(x_a, y_init, BC_t, t, U, L, obs_t, optimize_BC)
K_t_abs = K_t/x_a

## -------------------------------------------------------------------------##
# Prepare for scale factor adjustments
## -------------------------------------------------------------------------##
bootstrap_n = int(5e4)
sf = np.arange(-1.5, 1.6, 0.1)
sa_effect = np.zeros((len(sf), 4, bootstrap_n))
so_effect = np.zeros((len(sf), 4, bootstrap_n))

## -------------------------------------------------------------------------##
# Bootstrap through different random states
## -------------------------------------------------------------------------##
for rn in range(1, bootstrap_n):
    if rn % 1000 == 0:
        print(rn/bootstrap_n*100, '% complete')

    # Initialize the random state
    rs = RandomState(rn)

    # Define the prior
    x_a = np.abs(rs.normal(loc=70, scale=40, size=(nstate,))/(3600*24)) # prior (ppb/s)

    # Define the prior error
    s_a_vec = 0.5*x_a.mean()/x_a
    s_a_vec[s_a_vec < 0.5] = 0.5
    s_a_vec **= 2

    # Calculate the Jacobian
    K_t = K_t_abs*x_a

    ## ---------------------------------------------------------------------##
    # Iterate through perturbations to Sa and So
    ## ---------------------------------------------------------------------##
    for ind, i in enumerate(sf):
        # Alter Sa
        sa = (10**(2*i))*copy.deepcopy(s_a_vec)
        g = inv.get_gain_matrix(sa, s_o_vec, K_t, optimize_BC=False)
        g = g*x_a.reshape(-1, 1)*3600*24
        sa_effect[ind, 0, rn] = inv.band_width(g)
        sa_effect[ind, 2, rn] = inv.influence_length(g)

        sa = copy.deepcopy(s_a_vec)
        sa[0] *= (10**(2*i))
        g = inv.get_gain_matrix(sa, s_o_vec, K_t, optimize_BC=False)
        g = g*x_a.reshape(-1, 1)*3600*24
        sa_effect[ind, 1, rn] = inv.band_width(g)
        sa_effect[ind, 3, rn] = inv.influence_length(g)

        # Alter So
        so = (10**(2*i))*copy.deepcopy(s_o_vec)
        g = inv.get_gain_matrix(s_a_vec, so, K_t, optimize_BC=False)
        g = g*x_a.reshape(-1, 1)*3600*24
        so_effect[ind, 0, rn] = inv.band_width(g)
        so_effect[ind, 2, rn] = inv.influence_length(g)

        so = copy.deepcopy(s_o_vec)
        so[:nobs_per_cell] *= (10**(2*i))
        g = inv.get_gain_matrix(s_a_vec, so, K_t, optimize_BC=False)
        g = g*x_a.reshape(-1, 1)*3600*24
        so_effect[ind, 1, rn] = inv.band_width(g)
        so_effect[ind, 3, rn] = inv.influence_length(g)

## -------------------------------------------------------------------------##
# Average bootstrapped results
## -------------------------------------------------------------------------##
so_effect_sd = np.std(so_effect, axis=2)
so_effect = np.mean(so_effect, axis=2)

print(so_effect)
print(so_effect_sd)

sa_effect_sd = np.std(sa_effect, axis=2)
sa_effect = np.mean(sa_effect, axis=2)

## -------------------------------------------------------------------------##
# Plot
## -------------------------------------------------------------------------##
# ls = ['Lifetime', 'Prior error', 'Observational error']
# Iterate through band width and influence length scales
suffix = ['bw', 'ils']
yaxis = ['Gain matrix band width', 'Influence length scale']
ylim = [(220, 320), (220, 320), (0.5, 13.5), (0.5, 13.5)]
fig_summ, ax_summ = fp.get_figax(aspect=2, rows=2, cols=2,
                                 sharex=True, sharey=True)
# plt.subplots_adjust(wspace=0.5)
# We want 0 and 1 to --> 1 and 2 and 3 --> 2
for i, ax in enumerate(ax_summ.flatten()):
    ax.axvline(1, c='grey', zorder=10, ls='--', label='Base inversion')
    ax.plot(10**sf, sa_effect[:, i], c=fp.color(8), label='Prior error')
    ax.fill_between(10**sf,
                    sa_effect[:, i] - sa_effect_sd[:, i],
                    sa_effect[:, i] + sa_effect_sd[:, i],
                    color=fp.color(8), alpha=0.2, zorder=-1)
    ax.plot(10**sf, so_effect[:, i], c=fp.color(5), label='Observational error')
    ax.fill_between(10**sf,
                    so_effect[:, i] - so_effect_sd[:, i],
                    so_effect[:, i] + so_effect_sd[:, i],
                    color=fp.color(5), alpha=0.2, zorder=-1)
    ax.set_ylim(ylim[i])
    ax.set_xscale('log')

# Titles
ax_summ[0, 0] = fp.add_title(ax_summ[0, 0], 'Full domain scaled')
ax_summ[0, 1] = fp.add_title(ax_summ[0, 1], 'First grid cell scaled')

# Axis labels
ax_summ[0, 0] = fp.add_labels(ax_summ[0, 0], '', 'Gain matrix\nband width')
ax_summ[0, 1] = fp.add_labels(ax_summ[0, 1], '', '')
ax_summ[1, 0] = fp.add_labels(ax_summ[1, 0], 'Scale factor',
                              'Influence\nlength scale')
ax_summ[1, 1] = fp.add_labels(ax_summ[1, 1], 'Scale factor', '')

# Legend
handles_0, labels_0 = ax_summ[0, 0].get_legend_handles_labels()
ax_summ[0, 0] = fp.add_legend(ax_summ[0, 0], handles=handles_0, labels=labels_0,
                           bbox_to_anchor=(0.5, -0.1),
                           loc='upper center', ncol=4,
                           bbox_transform=fig_summ.transFigure)

fp.save_fig(fig_summ, plot_dir, f'g_summary')
