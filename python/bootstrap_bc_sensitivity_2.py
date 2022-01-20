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
bootstrap_n = int(5e4)#int(5e4)
perts = [25, 50, 100]
x_hat_effect = np.zeros((len(perts), nstate, bootstrap_n))
x_hat_effect_mod = np.zeros((len(perts), nstate, bootstrap_n))
g_sum_effect = np.zeros((len(perts), nstate, bootstrap_n))
g_sum_effect_mod = np.zeros((len(perts), nstate, bootstrap_n))

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

    # Calculate the true posterior
    y_a = fm.forward_model(x_a, y_init, BC_t, t, U, L, obs_t)
    inv_inputs = [x_a, s_a_vec, y.flatten(), y_a.flatten(), s_o_vec, K_t]
    x_hat_t, _, _, _ = inv.solve_inversion(*inv_inputs, optimize_BC)

    ## ---------------------------------------------------------------------##
    # Iterate through perturbations to the boundary condition
    ## ---------------------------------------------------------------------##
    for i, pert in enumerate(perts):
        BC = BC_t + pert

        # Solve inversion
        y_a = fm.forward_model(x_a, y_init, BC, t, U, L, obs_t)
        inv_inputs = [x_a, s_a_vec, y.flatten(), y_a.flatten(), s_o_vec, K_t]
        x_hat, s_hat, a, g = inv.solve_inversion(*inv_inputs, optimize_BC)
        x_hat_diff = np.abs(x_hat - x_hat_t)*x_a*3600*24

        # Solve inversion with modified So in first grid box
        s_o_vec_mod = copy.deepcopy(s_o_vec)
        s_o_vec_mod[:nobs_per_cell] *= (10**(2*0.5))
        inv_inputs = [x_a, s_a_vec,
                      y.flatten(), y_a.flatten(), s_o_vec_mod, K_t]
        x_hat_mod, s_hat_mod, a_mod, g_mod = inv.solve_inversion(*inv_inputs,
                                                                 optimize_BC)
        x_hat_diff_mod = np.abs(x_hat_mod - x_hat_t)*x_a*3600*24

        # Save out
        x_hat_effect[i, :, rn] = x_hat_diff
        g_sum_effect[i, :, rn] = np.abs(-pert*g.sum(axis=1))*x_a*3600*24
        x_hat_effect_mod[i, :, rn] = x_hat_diff_mod
        g_sum_effect_mod[i, :, rn] = np.abs(-pert*g_mod.sum(axis=1))*x_a*3600*24

## -------------------------------------------------------------------------##
# Average bootstrapped results
## -------------------------------------------------------------------------##
x_hat_effect_sd = np.std(x_hat_effect, axis=2)
x_hat_effect = np.mean(x_hat_effect, axis=2)

g_sum_effect_sd = np.std(g_sum_effect, axis=2)
g_sum_effect = np.mean(g_sum_effect, axis=2)

x_hat_effect_mod_sd = np.std(x_hat_effect_mod, axis=2)
x_hat_effect_mod = np.mean(x_hat_effect_mod, axis=2)

g_sum_effect_mod_sd = np.std(g_sum_effect_mod, axis=2)
g_sum_effect_mod = np.mean(g_sum_effect_mod, axis=2)

## -------------------------------------------------------------------------##
# Plot
## -------------------------------------------------------------------------##
fig_summ, ax_summ = plot.format_plot(nstate)
xp = np.arange(1, nstate+1)
for i, pert in enumerate(perts):
    ax_summ.plot(xp, x_hat_effect[i, :], c=fp.color(k=4*i), lw=1, ls='-',
                 label=f'+/-{pert:d} ppb')
    ax_summ.fill_between(xp,
                         x_hat_effect[i, :] - x_hat_effect_sd[i, :],
                         x_hat_effect[i, :] + x_hat_effect_sd[i, :],
                         color=fp.color(k=4*i), alpha=0.2, zorder=1)
    ax_summ.plot(xp, g_sum_effect[i, :], c=fp.color(k=4*i), lw=2, ls='--')
    ax_summ.plot(xp, g_sum_effect[i, :] - g_sum_effect_sd[i, :],
                 c=fp.color(k=4*i), lw=1, ls='--')
    ax_summ.plot(xp, g_sum_effect[i, :] + g_sum_effect_sd[i, :],
                 c=fp.color(k=4*i), lw=1, ls='--')

    # Modified
    ax_summ.plot(xp, x_hat_effect_mod[i, :], c=fp.color(k=4*i), lw=1, ls=':')
    ax_summ.fill_between(xp,
                         x_hat_effect_mod[i, :] - x_hat_effect_mod_sd[i, :],
                         x_hat_effect_mod[i, :] + x_hat_effect_mod_sd[i, :],
                         color=fp.color(k=4*i), alpha=0.2, zorder=1)

# Add ~10% errors
ax_summ.fill_between([0, nstate+1], 10, color='grey', alpha=0.2,
                     label=r'$\approx$ 10\% error')

# Add text
fp.add_title(ax_summ, 'Constant Boundary Condition Perturbations')
plot.add_text_label(ax_summ, optimize_BC)
fp.add_labels(ax_summ, 'State vector element',
              r'$\vert\Delta\hat{x}\vert$ (ppb/day)')

# Legend for summary plot
custom_lines = [Line2D([0], [0], color='grey', lw=1, ls='-'),
                Line2D([0], [0], color='grey', lw=2, ls='--'),
                Line2D([0], [0], color='grey', lw=1, ls=':')]
custom_labels = ['Numerical solution', 'Predicted solution',
                 'Modified observational error']
handles, labels = ax_summ.get_legend_handles_labels()
custom_lines.extend(handles)
custom_labels.extend(labels)
fp.add_legend(ax_summ, handles=custom_lines, labels=custom_labels,
              bbox_to_anchor=(0.5, -0.45), loc='upper center', ncol=3)

# Set limits
ax_summ.set_ylim(0, 100)

# Save plot
fp.save_fig(fig_summ, plot_dir, f'perturbation_summary')
plt.close()

