#%%
import numpy as np
import pandas as pd
from copy import deepcopy as dc
from matplotlib.lines import Line2D

# Custom packages
from utilities import plot_settings as ps
from utilities import inversion as inv
from utilities import format_plots as fp
from utilities import utils

project_dir, config = utils.setup()
data_dir = f'{project_dir}/data/data_OSSE'
plot_dir = f'{project_dir}/plots'

# rcParams['text.usetex'] = True
np.set_printoptions(precision=2, linewidth=300, suppress=True)

## -------------------------------------------------------------------------##
# Get the true inversion
## -------------------------------------------------------------------------##
# U = np.concatenate([np.arange(5, 0, -1), 
#                     np.arange(1, 5, 1)])*24
# U = np.repeat(U, 2)
# # U = 5*24

# if type(U) in [float, int]:
#     suffix = 'constwind'
# else:
#     suffix = 'varwind'

true_BC = inv.Inversion(gamma=1)
true_BC_opt = inv.Inversion(gamma=1, opt_BC=True)

## -------------------------------------------------------------------------##
# Define mean error tests
## -------------------------------------------------------------------------##
def get_err(xhat, norm=None, truth=None):
    # print(norm.mean())
    if truth is None:
        return (xhat**2).mean()**0.5/norm.mean()
    else:
        return ((xhat - truth)**2).mean()**0.5/norm.mean()

def linear_model(inv_obj, true_obj, buffer=False):
    if buffer:
        first_idx = 1
    else:
        first_idx = 0
    diff_true = (inv_obj.xhat - true_obj.xhat) # We normalize by xa, but xa == 1
    slope = (diff_true[-1] - diff_true[first_idx])/(inv_obj.nstate - 1)
    diff_model = slope*np.arange(inv_obj.nstate) + diff_true[first_idx]
    return diff_model

def agg_err(inv_obj, true_obj, buffer=False):
    if buffer:
        idx = 1
    else:
        idx = 0
    diff_abs = np.abs((inv_obj.xhat - true_obj.xhat)*inv_obj.xa_abs)[idx:].sum()
    post_abs = (true_obj.xhat*true_obj.xa_abs)[idx:].sum()
    return diff_abs/post_abs

def agg_err_frac(inv_obj, true_obj, frac=0.75):
    # if buffer:
    #     idx = 1
    # else:
    #     idx = 0
    diff_abs = np.abs((inv_obj.xhat - true_obj.xhat)*inv_obj.xa_abs)
    frac_err = np.cumsum(diff_abs)/diff_abs.sum()
    print(frac_err)
    idx = np.argwhere(frac_err > frac)[0][0]
    return idx

def ils(inv_obj, true_obj, frac=0.25):
    if inv_obj.buffer:
        idx = 1
    else:
        idx = 0
    diff_rel = np.abs((inv_obj.xhat - true_obj.xhat)/
                      inv_obj.xa[idx:inv_obj.nstate]) # xa = 1
    idx = _ils(diff_rel, frac)
    return idx

def _ils(diff_rel, frac=0.25):
    try:
        return np.argwhere(diff_rel <= frac)[0][0]
    except: 
        return len(diff_rel) + 1
    

def pert_base(inv_obj, true_obj, method, out, j, k, l, m):
    # diff = (inv_obj.xhat - true_obj.xhat)/inv_obj.xa
    # lin = linear_model(inv_obj, true_obj)
    if method == 'buffer':
        buffer = True
    else:
        buffer = False

    out[method]['err_ils'][j, k, l, m] = ils(
        inv_obj, true_obj)
    
    if out[method]['err_ils'][j, k, l, m] >= inv_obj.nstate:
        print(method, j)

def pert_standard(inv_obj, true_obj, pert, out, j, k, l, m):
    # Standard error metrics
    pert_base(inv_obj, true_obj, 'standard', out, j, k, l, m)

    # Preview and diagnostic
    prev, R = inv_obj.preview_2d(pert)
    prev = np.abs(prev)
    diag = np.abs(-pert*inv_obj.g.sum(axis=1)/inv_obj.xa)

    # Save out
    out['standard']['R'][j, k, l, m] = R
    out['standard']['prev'][j, k, l, m] = get_err(prev, inv_obj.xa)
    out['standard']['diag'][j, k, l, m] = get_err(diag, inv_obj.xa)
    out['standard']['prev_ils'][j, k, l, m] = _ils(prev)
    out['standard']['diag_ils'][j, k, l, m] = _ils(diag)

def make_metric_dict(nprior, nobs, nU, metrics=True):
    dic = {
        'err_ils' : np.zeros((nsamples, nprior, nU, nobs)),
        'R' : np.zeros((nsamples, nprior, nU, nobs)),
    }
    if metrics:
        met = {
            'prev_ils' :  np.zeros((nsamples, nprior, nU, nobs)),
            'diag_ils' : np.zeros((nsamples, nprior, nU, nobs)),
            'prev' :  np.zeros((nsamples, nprior, nU, nobs)),
            'diag' : np.zeros((nsamples, nprior, nU, nobs)),
        }
        dic.update(met)
    return dic
## -------------------------------------------------------------------------##
## Varying inversion parameters
## -------------------------------------------------------------------------##
#%%
nsamples = int(100) # Increase this later
nobs_per_cell = np.append(25, np.arange(50, 250, 50))
priors = np.sort(np.append(25, np.arange(20, 55, 7.5)))
Us = np.arange(3, 11, 1)
# wind = np.concatenate([np.arange(7, 3, -1), np.arange(3, 7, 1)])*24*60*60/1000

var_params = make_metric_dict(len(priors), len(nobs_per_cell), len(Us))
short_var_params = make_metric_dict(len(priors), len(nobs_per_cell), len(Us), 
                                    metrics=False)
err = {
    'standard' : dc(var_params), 
    'boundary' : dc(short_var_params), 
    'buffer' : dc(short_var_params), 
}
#%%

# In all cases, use a 10 ppb perturbation
pert = 10

for j in range(nsamples):
    if j % 10 == 0:
        print(j, '/', nsamples)
    j = int(j)
    # Sa variations
    for k, xa in enumerate(priors):
        # Get the prior
        xa_abs = np.random.RandomState(j).normal(
            loc=xa, scale=5, size=(true_BC.nstate,))
        sa = np.abs(xa_abs - true_BC.xt_abs)/true_BC.xt_abs
        sa = float(np.max(sa))
        for l, U in enumerate(Us):
            # Define the wind speed
            U = np.concatenate([np.arange(U + 2, U - 2, -1), 
                                np.arange(U - 2, U + 2, 1)])
            U = U*24*60*60/1000  
            for m, nobs in enumerate(nobs_per_cell):

                # Standard inversion
                inv_std = inv.Inversion(
                    rs=j, 
                    xa_abs=xa_abs, 
                    sa=sa, 
                    nobs_per_cell=nobs, 
                    U=U,
                    gamma=1, 
                    BC=true_BC.BCt
                )

                # Perturb the boundary condition and calculate the error
                inv_pert = inv.Inversion(
                    rs=j, 
                    xa_abs=xa_abs, 
                    sa=sa, 
                    nobs_per_cell=nobs, 
                    U=U,
                    gamma=1, 
                    BC=true_BC.BCt + pert
                )
                pert_standard(inv_pert, inv_std, pert, err, j, k, l, m)

                # BC correction
                inv_opt_BC = inv.Inversion(
                    rs=j, 
                    xa_abs=xa_abs, 
                    sa=sa, 
                    nobs_per_cell=nobs, 
                    U=U,
                    gamma=1, 
                    BC=true_BC.BCt + pert,
                    opt_BC=True
                )
                pert_base(inv_opt_BC, inv_std, 'boundary', err, j, k, l, m)
        
                # Buffer grid cell
                xa_abs_buff = np.random.RandomState(j).normal(
                    loc=xa, scale=5, size=(1,))
                inv_buff = inv.Inversion(
                    rs=j, 
                    xa_abs=np.append(xa_abs_buff, xa_abs), 
                    nobs_per_cell=nobs,
                    U=U,
                    gamma=1, 
                    BC=true_BC.BCt + pert,
                    buffer=True
                )
                pert_base(inv_buff, inv_std, 'buffer', err, j, k, l, m)

#%%
## -------------------------------------------------------------------------##
## Plot output
## -------------------------------------------------------------------------##
fig, ax = fp.get_figax(rows=1, cols=2, aspect=1.5, 
                        max_height=4.5, max_width=6,
                        sharex=False, sharey=True)
                    #    sharex=True, sharey=True)
ax = ax.flatten()
# plt.subplots_adjust(wspace=0.5, hspace=0.5)
# ax[0].set_ylim(0, 1.2)

xs = {'prior' : priors, 'nobs' : nobs_per_cell, 'U' : Us}
ls = ['-', '--', '--', ':', ':', ':']
lw = [5, 3, 3, 4, 4, 4]

colors = ['grey', # standard
          fp.color(3, 'viridis'), # boundary
          fp.color(7, 'plasma'), # buffer
          fp.color(4, 'plasma'), # 2D preview
          fp.color(7, 'plasma')] # diagnostic]
labels = ['No correction method', 'Boundary method',
          'Buffer method', 'Preview', 'Diagnostic']
Us_plot = np.tile(Us[None, None, :, None], 
                  (nsamples, len(priors), 1, len(nobs_per_cell)))*60*60/1000
for j, intervention in enumerate(['standard', 'boundary', 'buffer']):
    # Combine the sa and so modifications
    print(intervention)
    data = err[intervention]

    # Reduce R by the number of obs per grid cell so that it better
    # reflects the actual observational uncertainty
    x_r = (1/err['standard']['R']).flatten()
    y_r = data[f'err_ils'].flatten()

    idx_r = np.argsort(x_r)
    x_r = x_r[idx_r]
    y_r = y_r[idx_r]

    # ax[0].scatter(x_r, y_r, color=colors[j], s=5, alpha=0.5, 
    #               label=labels[j])

    x_r = np.array([arr.mean() for arr in np.split(x_r, 10)])
    y_r_mean = np.array([arr.mean() for arr in np.split(y_r, 10)])
    y_r_std = np.array([arr.std() for arr in np.split(y_r, 10)])

    ax[0].plot(x_r, y_r_mean, 
               color=colors[j], ls=ls[j], lw=lw[j], 
               label=labels[j])
    ax[0].fill_between(x_r, y_r_mean - y_r_std, y_r_mean + y_r_std, 
                         color=colors[j], alpha=0.25)

    # Wind speed
    x_u = (true_BC.L/Us_plot[:, 1, :, 1]).flatten()
    y_u = data[f'err_ils'][:, 1, :, 1].flatten()
    data_u = pd.DataFrame({'x_u' : x_u, 'y_u' : y_u})
    data_u = data_u.groupby('x_u').agg(['mean', 'std'])

    idx_u = np.argsort(x_u)
    x_u = x_u[idx_u]
    y_u = y_u[idx_u]

    # ax[1].scatter(x_u, y_u, color=colors[j], s=5, alpha=0.5, 
    #               label=labels[j])

    x_u = np.array([arr.mean() for arr in np.split(x_u, 10)])
    y_u_mean = np.array([arr.mean() for arr in np.split(y_u, 10)])
    y_u_std = np.array([arr.std() for arr in np.split(y_u, 10)])

    if intervention == 'buffer':
        tmp_y = err['buffer']['err_ils'][:, 1, :, 1].flatten()
        tmp_y = tmp_y[idx_u]
        print(tmp_y)
        print(y_u)
        tmp_y = np.array([arr.std() for arr in np.split(tmp_y, 10)])
        print(tmp_y)
        print(y_u_std)
        # print(x_u)
        # print(y_u_std)
        # print(y_u_mean - y_u_std)
        # print(y_u_mean + y_u_std)
        # print(2*y_u_std)
    ax[1].plot(x_u, y_u_mean, 
               color=colors[j], ls=ls[j], lw=lw[j],
               label=labels[j])
    ax[1].fill_between(x_u, y_u_mean - y_u_std, y_u_mean + y_u_std, 
                       color=colors[j], alpha=0.25)

    # Diagnostics
    if intervention == 'standard':
        y_r = data[f'prev_ils'].flatten()[idx_r]
        y_r_mean = np.array([arr.mean() for arr in np.split(y_r, 10)])
        y_r_std = np.array([arr.std() for arr in np.split(y_r, 10)])
        ax[0].plot(x_r, y_r_mean,
                   color=colors[3], ls=ls[3], lw=lw[3],
                   label=labels[3])
        ax[0].fill_between(x_r, y_r_mean - y_r_std, y_r_mean + y_r_std,
                           color=colors[3], alpha=0.25)
        
        y_u = data[f'prev_ils'][:, 1, :, 1].flatten()[idx_u]
        y_u_mean = np.array([arr.mean() for arr in np.split(y_u, 10)])
        y_u_std = np.array([arr.std() for arr in np.split(y_u, 10)])
        ax[1].plot(x_u, y_u_mean, color=colors[3], ls=ls[3], lw=lw[3])
        ax[1].fill_between(x_u, y_u_mean - y_u_std, y_u_mean + y_u_std, 
                           color=colors[3], alpha=0.25)
                
for axis in ax:
    for line in range(6):
        axis.axhline(2*line, lw=0.5, color='0.9', zorder=-10)

ax[0].axvline(1, lw=1, color='0.5', ls='--')
ax[0].text(1.2, 11.8, r'$R = 1$', va='top', ha='left')

# Add legend
handles, labels = ax[0].get_legend_handles_labels()

# Add a blank handle and label 
blank_handle = [Line2D([0], [0], markersize=0, lw=0)]
blank_label = ['']
handles.extend(blank_handle)
labels.extend(blank_label)

# Reorder
# reorder = [-1, 0, 1,
#            -1, 2, 3]
#         #    -1, 3, 4]
reorder = [1, -1, -1, 0, 2, 3]

handles = [handles[i] for i in reorder]
labels = [labels[i] for i in reorder]
# labels[3] = 'Error estimates : '
labels[2] = 'BC-induced error: '

ax[0].legend(handles=handles, labels=labels,
             loc='center left', alignment='center',  
             bbox_to_anchor=(0.95, 0.5), bbox_transform=fig.transFigure,
             ncol=1, handlelength=2, frameon=False, 
             fontsize=ps.LABEL_FONTSIZE*ps.SCALE) 

ax[0] = fp.add_title(ax[0], 'Varying error ratio')
ax[0] = fp.add_labels(
    ax[0], r'Information ratio $R$ [unitless]', 
    'Influence length scale [unitless]')
    # 'First grid cell index where\n'r'$\Delta \hat{x}/x_A \leq 0.25$ [unitless]')
ax[1] = fp.add_title(ax[1], 'Varying residence time')
ax[1] = fp.add_labels(
    ax[1], 
    'Residence 'r'time $\tau$ [hr]', '')

ax[0].text(0.025, 0.975, '(a)', fontsize=ps.LABEL_FONTSIZE*ps.SCALE,
           transform=ax[0].transAxes, ha='left', va='top')
ax[1].text(0.025, 0.975, '(b)', fontsize=ps.LABEL_FONTSIZE*ps.SCALE,
           transform=ax[1].transAxes, ha='left', va='top')

ax[0].set_xscale('log')
ax[0].set_xlim(0.075, 10)

ax[1].set_xlim(0.75, 2.25)
ax[0].set_ylim(0, 12)

# ax[2].set_ylim(0, 5)
# ax[3].set_ylim(0, 5)
fp.save_fig(fig, plot_dir, f'parameter_sensitivity')
# %%
