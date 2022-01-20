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

rcParams['text.usetex'] = True
np.set_printoptions(precision=1, linewidth=300, suppress=True)

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
# rs = RandomState(728)
rs = RandomState(625)

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
x_a = np.abs(rs.normal(loc=70, scale=40, size=(nstate,))/(3600*24)) # prior (ppb/s)
# x_a = 80*np.ones(nstate)/(3600*24)

# x_a = x_a - x_a.mean() + x_t.mean()
s_a_vec = 0.5*x_a.mean()/x_a
s_a_vec[s_a_vec < 0.5] = 0.5
# s_a_vec[:1] = 2
s_a_vec **= 2

# Define the observational errror
s_o_vec = 15*np.ones(nobs) #15
# s_o_vec[:1*nobs_per_cell] = 15*np.sqrt(10)
# s_o_vec[1*nobs_per_cell:2*nobs_per_cell] = 30
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
ax[0].plot(xp, 3600*24*x_a, #yerr=3600*24*x_a*s_a_vec**0.5,
            c=fp.color(4), marker='.', markersize=10, #capsize=2,
            label=r'Prior($\pm$$\approx$ 50\%)')
ax[0].fill_between(xp, 3600*24*x_a - 3600*24*x_a*s_a_vec**0.5,
                   3600*24*x_a + 3600*24*x_a*s_a_vec**0.5,
                   color=fp.color(2), alpha=0.2, zorder=-1)
handles_0, labels_0 = ax[0].get_legend_handles_labels()

ax[0] = fp.add_labels(ax[0], '', 'Emissions\n(ppb/day)')
ax[0].set_ylim(0, 200)

# Observations
ax[1].plot(xp, y_init, c='black', label='Steady state', zorder=10)
ax[1].plot(xp, y, c='grey', label='Observations\n($\pm$ 15 ppb)', lw=0.5,
           zorder=9)

# Error range
y_err_min = y.min(axis=1) - 15
y_err_max = y.max(axis=1) + 15
ax[1].fill_between(xp, y_err_min, y_err_max, color='grey', alpha=0.2)
                   # label=r'Error range ($\pm$ 15 ppb)')
handles_1, labels_1 = ax[1].get_legend_handles_labels()
handles_0.extend(handles_1)
labels_0.extend(labels_1)

# Aesthetics
ax[1] = fp.add_legend(ax[1], handles=handles_0, labels=labels_0,
                      bbox_to_anchor=(0.95, 0.5),
                      loc='center left', ncol=1,
                      bbox_transform=fig.transFigure)
ax[1] = fp.add_labels(ax[1], 'State vector element', 'XCH4\n(ppb)')
ax[1].set_ylim(y_init.min()-50, y_init.max()+50)

fp.save_fig(fig, plot_dir, f'prior_obs')

## -------------------------------------------------------------------------##
# Solve the inversion with the "true" boundary condition
## -------------------------------------------------------------------------##
# Inversion plots
xp = np.arange(1, nstate+1)
fig, ax = plot.format_plot(nstate)
ax = plot.plot_emis(ax, x_t, c=fp.color(2), ls='--', label='Truth')
ax = plot.plot_emis(ax, x_a, c=fp.color(4), marker='.',
                       markersize=10, label='Prior')

# Test 1: BC = truth
ls = ['--', '-']
labels = ['BC optimized', 'BC not optimized']
for i, optimize_BC in enumerate([False]):#enumerate([True, False]):
    y_a = fm.forward_model(x_a, y_init, BC_t, t, U, L, obs_t)
    K_t = inv.build_jacobian(x_a, y_init, BC_t, t, U, L, obs_t, optimize_BC)
    inv_inputs = [x_a, s_a_vec, y.flatten(), y_a.flatten(), s_o_vec, K_t]
    x_hat_t, s_hat, a_t, g_t = inv.solve_inversion(*inv_inputs, optimize_BC)
    y_hat_t = fm.forward_model(x_hat_t*x_a, y_init, BC_t, t, U, L, obs_t)
    bw_base = inv.band_width(g_t*x_a.reshape(-1, 1)*3600*24)
    ils_base = inv.influence_length(g_t*x_a.reshape(-1, 1)*3600*24)
    base = [bw_base, bw_base, ils_base, ils_base]

    ax = plot.plot_emis(ax, x_hat_t*x_a, c=fp.color(6), ls=ls[i], marker='*',
                        markersize=10, label=f'Posterior ({labels[i]})')

    # print('-'*300)
    # if optimize_BC:
    #     x_a_temp = np.append(x_a/j, [1900])
    # else:
    #     x_a_temp = x_a/j
    # for row in K_t:
    #     print(row/x_a_temp)
    # print('-'*300)

# Add legend
handles_0, labels_0 = ax.get_legend_handles_labels()
ax = fp.add_legend(ax, handles=handles_0, labels=labels_0,
                   bbox_to_anchor=(0.5, -0.45), loc='upper center', ncol=2)

# Add labels
ax = fp.add_labels(ax, 'State vector element', 'Emissions\n(ppb/day)')
ax = fp.add_title(ax, f'True Boundary Condition\n(BC = {BC_t:d} ppb)')
fp.save_fig(fig, plot_dir, f'constant_BC_{BC_t:d}')

## -------------------------------------------------------------------------##
# Compare Jacobian to theooretical Jacobian
## -------------------------------------------------------------------------##
# print('-'*300)
# for row in K_t:
#     print(row)
# print('-'*300)

# K_t_abs = K_t/x_a
# print(K_t.shape)
# print('-'*300)
# for row in K_t_abs:
#     print(row*j)
# print('-'*300)

# K_t_theory = np.tril(np.ones(K_t_abs.shape))
# print('-'*300)
# for row in K_t_theory:
#     print(row)
# print('-'*300)

# K_t_diff = j*K_t_abs - K_t_theory
# print('-'*300)
# for row in K_t_diff:
#     print(row*j)
# print('-'*300)

# fig, ax = fp.get_figax(aspect=1)
# ax.matshow(K_t_diff, cmap='RdBu_r', vmin=-1, vmax=1)
# fp.save_fig(fig, plot_dir, 'K_structure_diff')
# np.save('../data/K_t.npy', K_t_abs)
# np.save('../data/x_a.npy', x_a)
# np.save('../data/s_a_vec.npy', s_a_vec*x_a**2)

## -------------------------------------------------------------------------##
# Plot gain matrix for a number of So vectors
## -------------------------------------------------------------------------##
# x = np.arange(-1.5, 1.51, 0.01)
# sa_effect = np.zeros((len(x), 4))
# so_effect = np.zeros((len(x), 4))
# for ind, i in enumerate(x):
#     # Alter Sa
#     sa = (10**(2*i))*copy.deepcopy(s_a_vec)
#     g = inv.get_gain_matrix(sa, s_o_vec, K_t, optimize_BC=False)
#     g = g*x_a.reshape(-1, 1)*3600*24
#     sa_effect[ind, 0] = inv.band_width(g)
#     sa_effect[ind, 2] = inv.influence_length(g)

#     sa = copy.deepcopy(s_a_vec)
#     sa[0] *= (10**(2*i))
#     g = inv.get_gain_matrix(sa, s_o_vec, K_t, optimize_BC=False)
#     g = g*x_a.reshape(-1, 1)*3600*24
#     sa_effect[ind, 1] = inv.band_width(g)
#     sa_effect[ind, 3] = inv.influence_length(g)

#     # Alter So
#     so = (10**(2*i))*copy.deepcopy(s_o_vec)
#     g = inv.get_gain_matrix(s_a_vec, so, K_t, optimize_BC=False)
#     g = g*x_a.reshape(-1, 1)*3600*24
#     so_effect[ind, 0] = inv.band_width(g)
#     so_effect[ind, 2] = inv.influence_length(g)

#     so = copy.deepcopy(s_o_vec)
#     so[:nobs_per_cell] *= (10**(2*i))
#     g = inv.get_gain_matrix(s_a_vec, so, K_t, optimize_BC=False)
#     g = g*x_a.reshape(-1, 1)*3600*24
#     so_effect[ind, 1] = inv.band_width(g)
#     so_effect[ind, 3] = inv.influence_length(g)

# # Plotting
# # ls = ['Lifetime', 'Prior error', 'Observational error']
# # Iterate through band width and influence length scales
# suffix = ['bw', 'ils']
# yaxis = ['Gain matrix band width', 'Influence length scale']
# ylim = [(220, 310), (220, 310), (0.5, 10.5), (0.5, 10.5)]
# fig_summ, ax_summ = fp.get_figax(aspect=2, rows=2, cols=2,
#                                  sharex=True, sharey=True)
# # plt.subplots_adjust(wspace=0.5)
# # We want 0 and 1 to --> 1 and 2 and 3 --> 2
# for i, ax in enumerate(ax_summ.flatten()):
#     ax.scatter(1, base[i], c='grey', zorder=10, label='Base inversion')
#     ax.plot(10**x, sa_effect[:, i], c=fp.color(8), label='Prior error')
#     ax.plot(10**x, so_effect[:, i], c=fp.color(5), label='Observational error')
#     ax.set_ylim(ylim[i])
#     ax.set_xscale('log')

# # Titles
# ax_summ[0, 0] = fp.add_title(ax_summ[0, 0], 'Full domain scaled')
# ax_summ[0, 1] = fp.add_title(ax_summ[0, 1], 'First grid cell scaled')

# # Axis labels
# ax_summ[0, 0] = fp.add_labels(ax_summ[0, 0], '', 'Gain matrix\nband width')
# ax_summ[0, 1] = fp.add_labels(ax_summ[0, 1], '', '')
# ax_summ[1, 0] = fp.add_labels(ax_summ[1, 0], 'Scale factor',
#                               'Influence\nlength scale')
# ax_summ[1, 1] = fp.add_labels(ax_summ[1, 1], 'Scale factor', '')

# # Legend
# handles_0, labels_0 = ax_summ[0, 0].get_legend_handles_labels()
# ax_summ[0, 0] = fp.add_legend(ax_summ[0, 0], handles=handles_0, labels=labels_0,
#                            bbox_to_anchor=(0.5, -0.1),
#                            loc='upper center', ncol=4,
#                            bbox_transform=fig_summ.transFigure)

# fp.save_fig(fig_summ, plot_dir, f'g_summary')

## -------------------------------------------------------------------------##
# Constant boundary condition perturbations
## -------------------------------------------------------------------------##
# # Test 2: constant BC perturbation
# perts = [25, 50, 100]
# # fig_summ, ax_summ = format_plot(nstate, nplots=2, sharex=True)
# fig_summ, ax_summ = plot.format_plot(nstate)
# for i, pert in enumerate(perts):
#     BC = BC_t + pert

#     # Solve inversion
#     K = inv.build_jacobian(x_a, y_init, BC, t, U, L, obs_t, optimize_BC)
#     y_a = fm.forward_model(x_a, y_init, BC, t, U, L, obs_t)
#     inv_inputs = [x_a, s_a_vec, y.flatten(), y_a.flatten(), s_o_vec, K]
#     x_hat, s_hat, a, g = inv.solve_inversion(*inv_inputs, optimize_BC)
#     y_hat = fm.forward_model(x_hat*x_a, y_init, BC, t, U, L, obs_t)

#     # Plot inversion
#     fig, ax = plot.plot_inversion(x_a, x_hat, x_t, x_hat_true=x_hat_t,
#                               optimize_BC=optimize_BC)
#     ax = fp.add_title(ax, f'High Boundary Condition\n(BC = {BC:d} ppb)')
#     fp.save_fig(fig, plot_dir, f'constant_BC_{BC:d}')

#     # Plot observations
#     fig, ax = plot.plot_obs(nstate, y, y_a, y_init, obs_t, optimize_BC)
#     ax = fp.add_title(ax, f'High Boundary Condition\n(BC = {BC:d} ppb)')
#     fp.save_fig(fig, plot_dir, f'constant_BC_{BC:d}_obs')

#     fig, ax = plot.plot_obs_diff(nstate, y, y_hat, y_a, obs_t, optimize_BC)
#     ax = fp.add_title(ax, f'High Boundary Condition\n(BC = {BC:d} ppb)')
#     fp.save_fig(fig, plot_dir, f'constant_BC_{BC:d}_obs_diff')

#     # Plot cost function
#     fig, ax = plot.plot_cost_func(x_hat, x_a, s_a_vec, y_hat, y, s_o_vec,
#                                  obs_t, optimize_BC)
#     ax = fp.add_title(ax, f'Cost Function Components\n(BC = {BC:d} ppb)')
#     fp.save_fig(fig, plot_dir, f'constant_BC_{BC:d}_cost_func')

#     # Summary plot
#     ax_summ.plot(xp, np.abs(x_hat - x_hat_t)*x_a*3600*24,
#                     c=fp.color(k=4*i), lw=1, ls='-',
#                     label=f'{pert:d}/-{pert:d} ppb')
#     ax_summ.plot(xp, np.abs(-pert*g.sum(axis=1))*x_a*3600*24,
#                     c=fp.color(k=4*i), lw=2, ls='--')

# # Add ~10% errors
# ax_summ.fill_between([0, nstate+1], 10, color='grey', alpha=0.2,
#                      label=r'$\approx$ 10\% error')

# # Add text
# fp.add_title(ax_summ, 'Constant Boundary Condition Perturbations')
# plot.add_text_label(ax_summ, optimize_BC)
# fp.add_labels(ax_summ, 'State vector element',
#               r'$\vert\Delta\hat{x}\vert$ (ppb/day)')

# # Legend for summary plot
# custom_lines = [Line2D([0], [0], color='grey', lw=1, ls='-'),
#                 Line2D([0], [0], color='grey', lw=2, ls='--')]
# custom_labels = ['Numerical solution', 'Predicted solution']
# handles, labels = ax_summ.get_legend_handles_labels()
# custom_lines.extend(handles)
# custom_labels.extend(labels)
# fp.add_legend(ax_summ, handles=custom_lines, labels=custom_labels,
#               bbox_to_anchor=(0.5, -0.45), loc='upper center', ncol=3)

# # Set limits
# ax_summ.set_ylim(0, 100)

# # Save plot
# fp.save_fig(fig_summ, plot_dir, f'constant_BC_summary')
# plt.close()

## -------------------------------------------------------------------------##
# Oscillating boundary condition perturbations
## -------------------------------------------------------------------------##
optimize_BC = True
# Summary plot
fig_perts, ax_perts = fp.get_figax(aspect=5)
fig_summ, ax_summ = plot.format_plot(nstate)
# # Add ~10% errors
ax_summ.fill_between([0, nstate+1], 10, color='grey', alpha=0.2,
                     label=r'$\approx$ 10\% error')

# Iteration through perturbations
# BC = [vertical_shift, amplitude, frequency, phase shift]
# (pi halves should be a cosine)
BC1 = [1975, -100, 2, 0]
BC2 = [1975, 50, 2, 0]
BC3 = [1975, 100, 2, 0]
BC4 = [1975, -100, 2, np.pi/2]
BC5 = [1975, -100, 1, 0]
BC6 = [1975, -100, 4, 0]
# BC5 = [1975, -100, 6]
# # BC7 = [1975, -50, 2]
BCs = [BC1, BC2, BC3, BC4, BC5, BC6]
# BCs = [BC5]
# BCs = [BC1]
ls = ['--', '-']
x_hat_diff_record = np.zeros((2, len(BCs), nstate))
for i, BC_l in enumerate(BCs):
    for j, optimize_BC in enumerate([True, False]):
    # for i, BC_l in enumerate([BC1]):
        n = len(BCs)
        BC = BC_l[0] + BC_l[1]*np.sin(BC_l[2]*2*np.pi/t.max()*t+BC_l[3])

        # Label
        pi_unicode = "\u03C0"
        alpha_unicode = "\u03B1"
        if BC_l[3] > 0:
            label = r'%d + %d sin(%d$\alpha$t + $\pi$/2)' % (int(BC_l[0]), int(BC_l[1]), int(BC_l[2]))
        else:
            label = r'%d + %d sin(%d$\alpha$t)' % (int(BC_l[0]), int(BC_l[1]), int(BC_l[2]))

        # Solve inversion
        K = inv.build_jacobian(x_a, y_init, BC, t, U, L, obs_t, optimize_BC)
        y_a = fm.forward_model(x_a, y_init, BC, t, U, L, obs_t)
        inv_inputs = [x_a, s_a_vec, y.flatten(), y_a.flatten(), s_o_vec, K]
        x_hat, s_hat, a, g = inv.solve_inversion(*inv_inputs, optimize_BC)
        x_hat_diff = np.abs(x_hat - x_hat_t)*x_a*3600*24
        x_hat_diff_record[j, i, :] = x_hat_diff
        y_hat = fm.forward_model(x_hat*x_a, y_init, BC, t, U, L, obs_t)

        c = y_a.flatten() - K @ np.ones(K.shape[1])
        print('-'*70)
        print(np.abs(g.sum(axis=1))*x_a*3600*24)
        print(c)
        c = c.reshape(y_a.shape)
        # print(c)

        # Plots
        title = f'Oscillating Boundary Condition\n(BC = {int(BC_l[0]):d} + {int(BC_l[1]):d}sin({int(BC_l[2]):d}at) ppb)'
        # Perturbation plot
        if j == 0:
            ax_perts.plot(t/3600, BC, c=fp.color(i*2, lut=n*2), lw=2,
                          label=label)
        # print(BC)
        # if len(BCs) == 1:
        #     for j, c_row in enumerate(c):
        #         # print(c_row)
        #         if (j == 0) or (j == (nstate - 1)):
        #             jidx = j+1
        #             ax_perts.plot(obs_t/3600, c_row, #s=0.5+0.5*j, alpha=0.5,
        #                           c=fp.color(j*2, lut=c.shape[0]*2,
        #                                         cmap='Blues_r'),
        #                           lw=1, label=r'c(x$_{%d}$, t)' % jidx)
        #         else:
        #             ax_perts.plot(obs_t/3600, c_row, #s=0.5+0.5*j, alpha=0.5,
        #                           c=fp.color(j*2, lut=c.shape[0]*2,
        #                                         cmap='Blues_r'),
        #                           lw=1)
        #                              # lw=1-0.05*j,
        #                          # alpha=1-0.05*j)

        # for idx in range(9, 10):#, nstate+1):
        #     BC_test = (BC_l[0]
        #                + BC_l[1]*np.sin(BC_l[2]*2*np.pi/t.max()*(obs_t-idx*tau)))
        #     ax_perts.plot(obs_t/3600, BC_test, lw=0.5, ls='--',
        #                   c=fp.color(i*2, lut=n*2))
        # print()

        # # Inversion plot
        # fig, ax = plot.plot_inversion(x_a, x_hat, x_t, x_hat_t,
        #                               optimize_BC=optimize_BC)
        # ax = fp.add_title(ax, title)
        # fp.save_fig(fig, plot_dir, f'oscillating_BC_{(i+1)}_{optimize_BC}')

        # # Plot observations
        # fig, ax = plot.plot_obs(nstate, y, y_a, y_init, obs_t, optimize_BC)
        # ax = fp.add_title(ax, title)
        # fp.save_fig(fig, plot_dir, f'oscillating_BC_{(i+1)}_{optimize_BC}_obs')

        # fig, ax = plot.plot_obs_diff(nstate, y, y_hat, y_a, obs_t, optimize_BC)
        # ax = fp.add_title(ax, title)
        # fp.save_fig(fig, plot_dir, f'oscillating_BC_{(i+1)}_{optimize_BC}_obs_diff')

        # Summary plot
        ax_summ.plot(xp, x_hat_diff, c=fp.color(i*2, lut=n*2), lw=2, ls=ls[j],
                     label=label)

# styles = ['--', '-.', ':']
# # colors = ['0', '0.2', '0.4']
# perts = [25, 50, 100]
# for i, pert in enumerate(perts):
#     ax_summ.plot(xp, np.abs(-pert*g.sum(axis=1))*x_a*3600*24,
#                  c='black', lw=1, ls=styles[i],
#                  label=f'+/-{pert:d} ppb')
# ax_summ.fill_between(xp, np.sqrt(s_a_vec)*x_a*3600*24, color='grey', alpha=0.1,
#                      label='Prior error')
# ax_summ.fill_between(x_p, np.sqrt(np.diag(s_hat))*x_a*3600*24, color='grey', alpha=0.1)


# Perturbation summary aesthetics
ax_perts.axhline(BC_t, color='grey', ls='--', lw=2,
                 label='True boundary condition')
ax_perts.axvspan(obs_t.min()/3600, obs_t.max()/3600, color='grey', alpha=0.1,
                 label='Observation times')
ax_perts.set_xlim(t.min()/3600, t.max()/3600)
ax_perts.set_ylim(1700, 2300)
ax_perts = fp.add_title(ax_perts,
                        'Oscillating Boundary Condition Perturbations')
ax_perts = fp.add_labels(ax_perts, 'Time (hr)', 'BC (ppb)')
fp.add_legend(ax_perts, bbox_to_anchor=(0.5, -0.45), loc='upper center',
              ncol=2)
fp.save_fig(fig_perts, plot_dir, f'oscillating_BC_perts_summary_{optimize_BC}')

## Summary aesthetics
fp.add_title(ax_summ, 'Oscillating Boundary Condition Perturbations')
# plot.add_text_label(ax_summ, optimize_BC)
fp.add_labels(ax_summ, 'State vector element',
              r'$\vert\Delta\hat{x}\vert$ (ppb/day)')

# Set limits
ax_summ.set_ylim(0, 100)

# Add legend
custom_lines = [Line2D([0], [0], color='grey', lw=1, ls='-'),
                Line2D([0], [0], color='grey', lw=2, ls='--')]
custom_labels = ['BC not optimized', 'BC optimized']
handles, labels = ax_summ.get_legend_handles_labels()
custom_lines.extend(handles)
custom_labels.extend(labels)
fp.add_legend(ax_summ, handles=custom_lines, labels=custom_labels,
              bbox_to_anchor=(0.5, -0.45), loc='upper center', ncol=2)

# Save
fp.save_fig(fig_summ, plot_dir, f'oscillating_BC_summary_{optimize_BC}')

# One last plot
fig, ax = plot.format_plot(nstate)
for i, BC_l in enumerate(BCs):
    if BC_l[3] > 0:
        label = r'%d + %d sin(%d$\alpha$t + $\pi$/2)' % (int(BC_l[0]), int(BC_l[1]), int(BC_l[2]))
    else:
        label = r'%d + %d sin(%d$\alpha$t)' % (int(BC_l[0]), int(BC_l[1]), int(BC_l[2]))

    d = x_hat_diff_record[0, i, :] - x_hat_diff_record[1, i, :]
    ax.plot(xp, d, c=fp.color(i*2, lut=n*2), lw=2, ls=ls[j], label=label)
ax.set_ylim(-10, 10)
ax.axhline(0, ls='--', color='grey')
fp.save_fig(fig, plot_dir, 'optimize_BC_difference')


plt.close()
