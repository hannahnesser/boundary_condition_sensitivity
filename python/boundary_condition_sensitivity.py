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
# Define the model parameters
## -------------------------------------------------------------------------##
optimize_BC = False
print_summary = False

# Seed the random number generator
from numpy.random import RandomState
rs = RandomState(728)

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
# Generate pseudo-observations
## -------------------------------------------------------------------------##
# Create pseudo-observations
random_noise = rs.normal(0, 10, (nstate, nobs_per_cell))
y = y_init.reshape(-1,1) + random_noise

## -------------------------------------------------------------------------##
# Define the inversion parameters
## -------------------------------------------------------------------------##
# Define the prior and prior error
x_a = rs.normal(loc=70, scale=40, size=(nstate,))/(3600*24) # prior (ppb/s)
# x_a = x_a - x_a.mean() + x_t.mean()
s_a_vec = 0.5*x_a.mean()/x_a
s_a_vec[s_a_vec < 0.5] = 0.5
# s_a_vec[:2] = 1e9
s_a_vec **= 2

# Define the observational errror
s_o_vec = 15*np.ones(nobs) #15
# for i in range(0, nstate-1):
#     # print(f'{i}*nobs_per_cell : {(i+1)}*nobs_per_cell')
#     # print(15**(nstate-i))
#     s_o_vec[(i*nobs_per_cell):((i+1)*nobs_per_cell)] = (nstate - i)*15
# print(s_o_vec)
# s_o_vec[:1*nobs_per_cell] = 30
# s_o_vec[1*nobs_per_cell:2*nobs_per_cell] =
# s_o_vec[2*nobs_per_cell:3*nobs_per_cell] = 30
# s_o_vec[3*nobs_per_cell:4*nobs_per_cell] = 30
# s_o_vec[4*nobs_per_cell:5*nobs_per_cell] = 30
s_o_vec **= 2
# s_o = s_o_vec**2*np.identity(nobs)


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
# Set up plot directory
## -------------------------------------------------------------------------##
plot_dir = f'{plot_dir}/n{nstate}_m{nobs}'
if not os.path.exists(plot_dir):
    os.mkdir(plot_dir)

## -------------------------------------------------------------------------##
# Print information about the study
## -------------------------------------------------------------------------##
if print_summary:
    print('-'*20, 'BOUNDARY CONDITION SENSITIVITY STUDY', '-'*20)
    print(f'TIME STEP (hrs)          : {(delta_t/3600)}')
    print(f'GRID CELL LIFETIME (hrs) : {(tau/3600)}')
    print(f'TRUE EMISSIONS (ppb/hr)  :', x_t[0]*3600)
    print(f'STEADY STATE (ppb)       :', y_ss)
    print(f'OBSERVATION TIMES (hrs)  :', obs_t/3600)
    print(f'MODEL TIMES (hrs)        :', t/3600)
    print('PRIOR EMISSIONS           : ', x_a*3600)
    print('-'*78)

## -------------------------------------------------------------------------##
# Create initial plots
## -------------------------------------------------------------------------##
# Prior
fig, ax = plot.format_plot(nstate, nplots=2, sharex=True)
fp.add_title(ax[0], 'Base inversion variables')
xp = np.arange(1, nstate+1) # plotting x coordinates

# Plot "true " emissions
ax[0].plot(xp, 3600*24*x_t, c=fp.color(2), ls='--', label='Truth')
ax[0].errorbar(xp, 3600*24*x_a, yerr=3600*24*x_a*s_a_vec**0.5,
            c=fp.color(4), marker='.', markersize=10, capsize=2,
            label=r'Prior ($\pm$$\approx$ 50\%)')
handles_0, labels_0 = ax[0].get_legend_handles_labels()

ax[0] = fp.add_labels(ax[0], '', 'Emissions\n(ppb/day)')
ax[0].set_ylim(0, 200)

# Observations
ax[1].plot(xp, y_init, c='black', label='Steady state', zorder=10)
ax[1].plot(xp, y, c='grey', label='Observations ($\pm$ 15 ppb)',
           zorder=9, alpha=0.4)

# Error range
y_err_min = (y - 15).min(axis=1)
y_err_max = (y + 15).max(axis=1)
ax[1].fill_between(xp, y_err_min, y_err_max, color='grey', alpha=0.2)
                   # label=r'Error range ($\pm$ 15 ppb)')
handles_1, labels_1 = ax[1].get_legend_handles_labels()
handles_0.extend(handles_1)
labels_0.extend(labels_1)

# Aesthetics
ax[1] = fp.add_legend(ax[1], handles=handles_0, labels=labels_0,
                      bbox_to_anchor=(0.5, -0.35),
                      loc='upper center', ncol=2)
ax[1] = fp.add_labels(ax[1], 'State Vector Element', 'XCH4\n(ppb)')
ax[1].set_ylim(y_init.min()-50, y_init.max()+50)

fp.save_fig(fig, plot_dir, f'prior_obs')

## -------------------------------------------------------------------------##
# Solve the inversion with the "true" boundary condition
## -------------------------------------------------------------------------##
# Inversion plots
xp = np.arange(1, nstate+1)

# Test 1: BC = truth
y_a = fm.forward_model(x_a, y_init, BC_t, t, U, L, obs_t)
K_t = inv.build_jacobian(x_a, y_init, BC_t, t, U, L, obs_t, optimize_BC)
inv_inputs = [x_a, s_a_vec, y.flatten(), y_a.flatten(), s_o_vec, K_t]
x_hat_t, s_hat, a_t, g_t = inv.solve_inversion(*inv_inputs, optimize_BC)
y_hat_t = fm.forward_model(x_hat_t*x_a, y_init, BC_t, t, U, L, obs_t)

print(K_t)

fig, ax = plot.plot_inversion(x_a, x_hat_t, x_t, #s_a=s_a,
                              optimize_BC=optimize_BC)
ax = fp.add_title(ax, f'True Boundary Condition\n(BC = {BC_t:d} ppb)')
fp.save_fig(fig, plot_dir, f'constant_BC_{BC_t:d}')

fig, ax = plot.plot_avker(nstate, a_t)
if optimize_BC:
    title = 'Boundary Condition Optimized'
else:
    title = 'Boundary Condition Not Optimized'
ax = fp.add_title(ax, title)
fp.save_fig(fig, plot_dir, f'avker_optBC_{optimize_BC}')

fig, ax = plot.plot_obs_diff(nstate, y, y_hat_t, y_a, obs_t, optimize_BC)
ax = fp.add_title(ax, f'True Boundary Condition\n(BC = {BC_t:d} ppb)')
fp.save_fig(fig, plot_dir, f'constant_BC_{BC_t:d}_obs_diff')

fig, ax = plot.plot_cost_func(x_hat_t, x_a, s_a_vec, y_hat_t, y, s_o_vec,
                              obs_t, optimize_BC)
ax = fp.add_title(ax, f'Cost Function Components\n(BC = {BC_t:d} ppb)')
fp.save_fig(fig, plot_dir, f'constant_BC_{BC_t:d}_cost_func')
plt.close()

## -------------------------------------------------------------------------##
# Constant boundary condition perturbations
## -------------------------------------------------------------------------##
# Test 2: constant BC perturbation
perts = [50, 100, 200]
# fig_summ, ax_summ = format_plot(nstate, nplots=2, sharex=True)
fig_summ, ax_summ = plot.format_plot(nstate)
for i, pert in enumerate(perts):
    BC = BC_t + pert

    # Solve inversion
    K = inv.build_jacobian(x_a, y_init, BC, t, U, L, obs_t, optimize_BC)
    y_a = fm.forward_model(x_a, y_init, BC, t, U, L, obs_t)
    inv_inputs = [x_a, s_a_vec, y.flatten(), y_a.flatten(), s_o_vec, K]
    x_hat, s_hat, a, g = inv.solve_inversion(*inv_inputs, optimize_BC)
    y_hat = fm.forward_model(x_hat*x_a, y_init, BC, t, U, L, obs_t)

    # Plot inversion
    fig, ax = plot.plot_inversion(x_a, x_hat, x_t, x_hat_t,
                             optimize_BC=optimize_BC)
    ax = fp.add_title(ax, f'High Boundary Condition\n(BC = {BC:d} ppb)')
    fp.save_fig(fig, plot_dir, f'constant_BC_{BC:d}')

    # Plot observations
    fig, ax = plot.plot_obs(nstate, y, y_a, y_init, obs_t, optimize_BC)
    ax = fp.add_title(ax, f'High Boundary Condition\n(BC = {BC:d} ppb)')
    fp.save_fig(fig, plot_dir, f'constant_BC_{BC:d}_obs')

    fig, ax = plot.plot_obs_diff(nstate, y, y_hat, y_a, obs_t, optimize_BC)
    ax = fp.add_title(ax, f'High Boundary Condition\n(BC = {BC:d} ppb)')
    fp.save_fig(fig, plot_dir, f'constant_BC_{BC:d}_obs_diff')

    # Plot cost function
    fig, ax = plot.plot_cost_func(x_hat, x_a, s_a_vec, y_hat, y, s_o_vec,
                                 obs_t, optimize_BC)
    ax = fp.add_title(ax, f'Cost Function Components\n(BC = {BC:d} ppb)')
    fp.save_fig(fig, plot_dir, f'constant_BC_{BC:d}_cost_func')

    # Summary plot
    ax_summ.plot(xp, np.abs(x_hat - x_hat_t)*x_a*3600*24,
                    c=fp.color(k=4*i), lw=1, ls='-',
                    label=f'{pert:d}/-{pert:d} ppb')
    ax_summ.plot(xp, np.abs(-pert*g.sum(axis=1))*x_a*3600*24,
                    c=fp.color(k=4*i), lw=2, ls='--')

# Add text
fp.add_title(ax_summ, 'Constant Boundary Condition Perturbations')
plot.add_text(ax_summ, optimize_BC)
fp.add_labels(ax_summ, 'State Vector Element',
              r'$\vert\Delta\hat{x}\vert$ (ppb/day)')

# Legend for summary plot
custom_lines = [Line2D([0], [0], color='grey', lw=1, ls='-'),
                Line2D([0], [0], color='grey', lw=2, ls='--')]
custom_labels = ['Numerical solution', 'Predicted solution']
handles, labels = ax_summ.get_legend_handles_labels()
custom_lines.extend(handles)
custom_labels.extend(labels)
fp.add_legend(ax_summ, handles=custom_lines, labels=custom_labels,
              bbox_to_anchor=(0.5, -0.35), loc='upper center', ncol=3)

# Set limits
ax_summ.set_ylim(0, 200)

# Save plot
fp.save_fig(fig_summ, plot_dir, f'constant_BC_summary')
plt.close()

## -------------------------------------------------------------------------##
# Oscillating boundary condition perturbations
## -------------------------------------------------------------------------##

# # Test 3: oscillating BC perturbation
# # Summary plot
# fig_perts, ax_perts = fp.get_figax(aspect=5)
# fig_summ, ax_summ[0] = format_plot(nstate)
# # BC = [vertical_shift, amplitude, frequency]
# BC1 = [2000, -200, 1]
# BC2 = [2000, 200, 1]
# BC3 = [2000, -100, 1]
# BC4 = [2000, -200, 2]
# BC5 = [2000, -200, 3]
# BCs = [BC1, BC2, BC3, BC4, BC5]
# for i, BC_l in enumerate(BCs):
#     BC = BC_l[0] + BC_l[1]*np.sin(BC_l[2]*2*np.pi/t.max()*t)

#     # Solve inversion
#     K = inv.build_jacobian(x_a, y_init, BC, t, U, L, obs_t,
#                        optimize_BC)
#     y_a = fm.forward_model(x_a, y_init, BC, t, U, L, obs_t)
#     c = y_a.flatten() - K @ np.ones(K.shape[1])
#     x_hat, s_hat, a = inv.solve_inversion(x_a, s_a,
#                                       y.flatten(), y_a.flatten(), s_o, K,
#                                       optimize_BC, verbose=True)

#     # # Plots
#     # Perturbation plot
#     ax_perts.plot(t/3600, BC, c=fp.color(i*2, lut=len(BCs)*2), lw=2)


#     # # Inversion plot
#     # fig, ax = plot.plot_inversion(x_a, x_hat, x_t, x_hat_t,
#     #                          optimize_BC=optimize_BC)
#     # ax = fp.add_title(ax,
#     #                   f'Oscillating Boundary Condition\n(BC = {int(BC_l[0]):d} + {int(BC_l[1]):d}sin({int(BC_l[2]):d}at) ppb)')

#     # fp.save_fig(fig, plot_dir, f'oscillating_BC{i}')

#     fig, ax = plot.plot_obs_diff(nstate, y, y_hat, y_a, obs_t, optimize_BC)
#     ax.scatter(xp, c, c='red', s=30)
#     ax.scatter(xp, (K @ np.ones(K.shape[1])), c='blue', s=30)
#     print(BC)
#     print(c)

#     ax = fp.add_title(ax,
#                       f'Oscillating Boundary Condition\n(BC = {int(BC_l[0]):d} + {int(BC_l[1]):d}sin({int(BC_l[2]):d}at) ppb)')
#     fp.save_fig(fig, plot_dir, f'oscillating_BC{i}_obs')

#     # # Summary plot
#     # ax_summ.plot(xp, np.abs(x_hat - x_hat_t)*x_a*3600*24,
#     #              c=fp.color(i*2, lut=len(BCs)*2), lw=2, ls='-',
#     #              label=f'Test {i+1}')

# ax_perts.set_xlim(t.min()/3600, t.max()/3600)
# ax_perts.set_ylim(1600, 2300)
# ax_perts = fp.add_title(ax_perts, 'Oscillating Boundary Condition Perturbations')
# ax_perts = fp.add_labels(ax_perts, 'Time (hr)', 'BC (ppb)')
# fp.save_fig(fig_perts, plot_dir, f'oscillating_BC_perts_summary')

# if optimize_BC:
#     txt = 'BC optimized'
# else:
#     txt = 'BC not optimized'
# ax_summ.text(0.98, 0.95, txt, ha='right', va='top',
#              fontsize=config.LABEL_FONTSIZE*config.SCALE,
#              transform=ax_summ.transAxes)
# ax_summ.set_ylim(0, 100)

# fp.add_title(ax_summ, 'Oscillating Boundary Condition Perturbations')
# fp.add_labels(ax_summ, 'State Vector Element', r'$\Delta$XCH4 (ppb)')
# fp.add_legend(ax_summ, bbox_to_anchor=(0.5, -0.35), loc='upper center', ncol=3)
# fp.save_fig(fig_summ, plot_dir, f'oscillating_BC_summary')
# plt.close()
