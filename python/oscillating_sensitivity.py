import numpy as np
import pandas as pd
import copy
import os
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.lines import Line2D
import matplotlib.colors as colors

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

# Seed the random number generator
from numpy.random import RandomState
# rs = RandomState(728)
rs = RandomState(625)

# Define the parameters of our simple forward model
C = 0.5 # Courant number
L = 25 #12.5 # grid cell length (25 km)

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

## -------------------------------------------------------------------------##
# Define the inversion parameters
## -------------------------------------------------------------------------##
# Define the prior and prior error
x_a = np.abs(rs.normal(loc=70, scale=40, size=(nstate,))/(3600*24)) # prior (ppb/s)
# x_a = 80*np.ones(nstate)/(3600*24)

s_a_vec = 0.5*x_a.mean()/x_a
s_a_vec[s_a_vec < 0.5] = 0.5
s_a_vec **= 2

# Define the observational errror
s_o_vec = 15*np.ones(nobs) #15
s_o_vec **= 2

## -------------------------------------------------------------------------##
# Plotting materials
## -------------------------------------------------------------------------##
xp = np.arange(1, nstate+1) # plotting x coordinates

# Set up plot directory
plot_dir = f'{plot_dir}/n{nstate}_m{nobs}'
if not os.path.exists(plot_dir):
    os.mkdir(plot_dir)

## -------------------------------------------------------------------------##
# Iterate through wind speed and frequency of perturbations
# Eventually need to iterate through amplitude of errors
wind_speeds = np.array([5, 10])#np.arange(1, 10, 2)
freqs = np.arange(1, 8, 2)
lsls = ['-', '--', ':', '-.']
summ = pd.DataFrame(columns=['lifetime', 'freq', 'period', 'ils',
                             'in_ils_err', 'out_ils_err'])
fig_summ, ax_summ = plot.format_plot(nstate)

for k, u in enumerate(wind_speeds):
    U = u/3600 # windspeed (5 km/hr in km/s)
    j = U/L # transfer coeficient (s-1)
    tau = 1/j # lifetime

    # Define the initial conditions
    y_init = [BC_t + x_t[0]/j]
    for i in range(1, nstate):
        y_init.append(y_init[-1] + x_t[i]/j)
    y_init = np.array(y_init)
    y_init = copy.deepcopy(y_init)

    # Create pseudo-observations
    random_noise = rs.normal(0, 10, (nstate, nobs_per_cell))
    y = y_init.reshape(-1,1) + random_noise

    # Define the times at which we sample the forward model and the observations
    delta_t = C*L/U # seconds
    t = np.arange(0, init_t + total_t + delta_t, delta_t)
    obs_t = np.linspace(init_t + delta_t, init_t + total_t, nobs_per_cell)

    # Construct the Jacobian
    K_t = inv.build_jacobian(x_a, y_init, BC_t, t, U, L, obs_t, optimize_BC)

    # Generate the prior observations
    y_a = fm.forward_model(x_a, y_init, BC_t, t, U, L, obs_t)

    # Calculate c
    c_t = y_a.flatten() - K_t @ np.ones(K_t.shape[1])

    # Solve the true inversion
    inv_inputs = [x_a, s_a_vec, y.flatten(), y_a.flatten(), s_o_vec, K_t]
    x_hat_t, s_hat, a_t, g_t = inv.solve_inversion(*inv_inputs, optimize_BC)
    y_hat_t = fm.forward_model(x_hat_t*x_a, y_init, BC_t, t, U, L, obs_t)

    for l, f in enumerate(freqs):
        # for amp in [50, 75, 100]
        freq = f*2*np.pi/t.max()
        BC = [1900, 50, freq, 0]
        BC = BC[0] + BC[1]*np.sin(BC[2]*t + BC[3])

        # Solve inversion
        # Construct the Jacobian
        K = inv.build_jacobian(x_a, y_init, BC, t, U, L, obs_t, optimize_BC)

        # Generate the prior observations
        y_a = fm.forward_model(x_a, y_init, BC, t, U, L, obs_t)

        # Solve the inversion
        inv_inputs = [x_a, s_a_vec, y.flatten(), y_a.flatten(), s_o_vec, K]
        x_hat, s_hat, a, g = inv.solve_inversion(*inv_inputs, optimize_BC)
        x_hat_diff = np.abs(x_hat - x_hat_t)*x_a*3600*24

        # Calculate the ILS and the error withiin and outside of ILS
        ils = inv.influence_length(x_hat, x_hat_t, threshold=0.1)
        in_ils_err, out_ils_err = inv.xhat_err(x_hat,#*x_a*3600*24,
                                               x_hat_t,#*x_a*3600*24,
                                               ils)
        if out_ils_err > 0.1:
            print(f, u)

        # Save out summary
        summ_df = {'lifetime' : tau, 'freq' : freq, 'period' : (2*np.pi/freq),
                   'ils' : ils, 'in_ils_err' : in_ils_err,
                   'out_ils_err' : out_ils_err}
        summ = summ.append(summ_df, ignore_index=True)

        # Plots
        print('---', i, len(wind_speeds))
        ax_summ.plot(xp, x_hat_diff, c=fp.color(k*2, lut=len(wind_speeds)*2),
                     lw=1, ls=lsls[l],
                     label=f'{u} km/hr, {(2*np.pi/freq/3600/24):.2f} hr')

        # # Calculate the difference between the true and perturbed inversion
        # x_hat_diff = np.abs(x_hat - x_hat_t)*x_a*3600*24
        # x_hat_diff_record[j, i, :] = x_hat_diff
        # y_hat = fm.forward_model(x_hat*x_a, y_init, BC, t, U, L, obs_t)

        # c = y_a.flatten() - K @ np.ones(K.shape[1])
        # delta_c = c - c_t
        # x_hat_diff_pred = np.array(-(g*x_a.reshape((-1, 1))*3600*24) @ delta_c)

print(summ)


# Frequency
# ax[0, 0].scatter(summ['freq'], summ['ils'], c=summ['lifetime'], s=1,
#                  cmap='plasma')
# ax[1, 0].scatter(summ['freq'], summ['out_ils_err'], c=summ['lifetime'],
#                  cmap='plasma', marker='^', s=1)

# # Lifetime
# ax[0, 1].scatter(summ['lifetime'], summ['ils'], c=summ['freq'], s=1,
#                  cmap='plasma')
# ax[1, 1].scatter(summ['lifetime'], summ['out_ils_err'], c=summ['freq'],
#                  cmap='plasma', marker='^', s=1)

# Summary plot
ax_summ.set_ylim(0, 50)
ax_summ.axhspan(0, 10, color='grey', alpha=0.2,
              label=r'$\approx$ 10\% error')
fp.add_legend(ax_summ, bbox_to_anchor=(0.5, -0.45), loc='upper center', ncol=2)
fp.save_fig(fig_summ, plot_dir, f'oscillating_summary_xhatdiff')

# Dimensionless 1
fig, ax = fp.get_figax(aspect=2, rows=1, cols=2)
cax = fp.add_cax(fig, ax)
c = ax[0].scatter(summ['period']/summ['lifetime'], summ['ils'],
                  c=summ['period']/3600/24, cmap='plasma', s=1, label='ILS')
# ax[1].scatter(summ['freq']*summ['lifetime'], summ['in_ils_err'],
#             marker='x', color=fp.color(5), s=1, label='Error within ILS')
c = ax[1].scatter(summ['period']/summ['lifetime'], summ['out_ils_err'],
                  c=summ['period']/3600/24, cmap='plasma', marker='^', s=1,
                  label='Error outside of ILS')
cb = fig.colorbar(c, cax=cax)
cb = fp.format_cbar(cb, r'Period (h)')

# Aesthetics
ax[1].axhspan(0, 0.1, color='grey', alpha=0.2,
              label=r'$\approx$ 10\% error')
ax[1].set_ylim(0, 0.3)
# ax[1, 2] = fp.add_legend(ax[1, 2])

fp.save_fig(fig, plot_dir, 'oscillating_summary_ILS_1')

# Dimensionless 2
fig, ax = fp.get_figax(aspect=2, rows=1, cols=2)
cax = fp.add_cax(fig, ax)
c = ax[0].scatter(summ['period']/summ['lifetime'], summ['ils'],
                  c=summ['lifetime']/3600/24, cmap='plasma',
                  # norm=colors.LogNorm(vmin=1, vmax=0.05),
                  s=1, label='ILS')
# ax[1].scatter(summ['freq']*summ['lifetime'], summ['in_ils_err'],
#             marker='x', color=fp.color(5), s=1, label='Error within ILS')
c = ax[1].scatter(summ['period']/summ['lifetime'], summ['out_ils_err'],
                  c=summ['lifetime']/3600/24, cmap='plasma',
                  # norm=colors.LogNorm(vmin=1, vmax=0.05),
                  marker='^', s=1, label='Error outside of ILS')
cb = fig.colorbar(c, cax=cax)
cb = fp.format_cbar(cb, 'Lifetime (h)')

# Aesthetics
ax[1].axhspan(0, 0.1, color='grey', alpha=0.2,
              label=r'$\approx$ 10\% error')
ax[1].set_ylim(0, 0.3)
# ax[1, 2] = fp.add_legend(ax[1, 2])

fp.save_fig(fig, plot_dir, 'oscillating_summary_ILS_2')

print(summ['lifetime']/3600/24)



