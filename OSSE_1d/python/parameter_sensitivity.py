#%%
import numpy as np
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

def agg_err_frac(inv_obj, true_obj, frac=0.75, buffer=False):
    if buffer:
        idx = 1
    else:
        idx = 0
    diff_abs = np.abs((inv_obj.xhat - true_obj.xhat)*inv_obj.xa_abs)[idx:]
    frac_err = np.cumsum(diff_abs)/diff_abs.sum()
    print(frac_err)
    idx = np.argwhere(frac_err > frac)[0][0]
    return idx

def ils(inv_obj, true_obj, frac=0.25, buffer=False):
    if buffer:
        idx = 1
    else:
        idx = 0

    diff_rel = np.abs((inv_obj.xhat - true_obj.xhat)/
                      inv_obj.xa[:inv_obj.nstate])[idx:] # xa = 1
    idx = _ils(diff_rel, frac)
    return idx

def _ils(diff_rel, frac=0.25):
    try:
        return np.argwhere(diff_rel <= frac)[0][0]
    except: 
        return len(diff_rel) + 1
    

def pert_base(inv_obj, true_obj, method, vari, out, i, j):
    # diff = (inv_obj.xhat - true_obj.xhat)/inv_obj.xa
    # lin = linear_model(inv_obj, true_obj)
    if method == 'buffer':
        buffer = True
        start_idx = 1
    else:
        buffer = False
        start_idx = 0

    # out[method]['err'][i, j] = get_err(
    #     inv_obj.xhat[start_idx:], 
    #     inv_obj.xa[start_idx:inv_obj.nstate], 
    #     true_obj.xhat[start_idx:])
    # # out[method]['err_frac'][i, j] = agg_err_frac(
    # #     inv_obj, true_obj, buffer=buffer)
    # out[method]['err_agg'][i, j] = agg_err(
    #     inv_obj, true_obj, buffer=buffer)
    out[method]['err_ils'][i, j] = ils(
        inv_obj, true_obj, buffer=buffer)
    
    if out[method]['err_ils'][i, j] >= inv_obj.nstate:
        print(method, vari, j)

def pert_standard(inv_obj, true_obj, pert, vari, out, j, k, l, m):
    # Standard error metrics
    pert_base(inv_obj, true_obj, 'standard', vari, out, j, k, l, m)

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
    # out['standard']['prev_agg'][j, k, l, m] = agg_err(inv_obj, true_obj)
    # out['standard']['diag_agg'][j, k, l, m] = agg_err(inv_obj, true_obj)

def make_metric_dict(nprior, nobs, nU, metrics=True):
    dic = {
        # 'err' : np.zeros((nprior, nobs, nU, nsamples)),
        # # 'err_frac' : np.zeros((nprior, nobs, nU, nsamples)),
        # 'err_agg' : np.zeros((nprior, nobs, nU, nsamples)),
        'err_ils' : np.zeros((nprior, nobs, nU, nsamples)),
        'R' : np.zeros((nprior, nobs, nU, nsamples)),
    }
    if metrics:
        met = {
            'prev_ils' :  np.zeros((nprior, nobs, nU, nsamples)),
            'diag_ils' : np.zeros((nprior, nobs, nU, nsamples)),
            'prev_agg' :  np.zeros((nprior, nobs, nU, nsamples)),
            'diag_agg' : np.zeros((nprior, nobs, nU, nsamples)),
            'prev' :  np.zeros((nprior, nobs, nU, nsamples)),
            'diag' : np.zeros((nprior, nobs, nU, nsamples)),
        }
        dic.update(met)
    return dic
## -------------------------------------------------------------------------##
## Varying inversion parameters
## -------------------------------------------------------------------------##
#%%
nsamples = int(10) # Increase this later
nobs_per_cell = np.arange(25, 260, 50)
priors = np.arange(20, 50, 7.5)
# sas = np.arange(0.25, 2.25, 0.25)
# sos = np.append(1, np.arange(5, 25))
Us = np.array([2.5, 5, 7.5, 10])
wind = np.concatenate([np.arange(7, 3, -1), np.arange(3, 7, 1)])*24*60*60/1000

# var_params = {
#     'prior' : make_metric_dict(len(priors)),
#     'nobs' : make_metric_dict(len(nobs_per_cell)),
#     'U' : make_metric_dict(len(Us))
# }

# short_var_params = {
#     'prior' : make_metric_dict(len(priors), metrics=False),
#     'nobs' : make_metric_dict(len(nobs_per_cell), metrics=False),
#     'U' : make_metric_dict(len(Us), metrics=False)
# }

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
        sa = np.abs(true_BC.xt_abs - xa)/true_BC.xt_abs
        sa = float(np.max(sa))
        for l, U in enumerate(Us):
            # Define the wind speed
            U = np.concatenate([np.arange(U + 2, U - 2, -1), 
                                np.arange(U - 2, U + 2, 1)])
            U *= 24*60*60/1000  
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
                pert_standard(inv_pert, inv_std, pert, err, i, j)

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
                pert_base(inv_opt_BC, inv_std, 'boundary', 'prior', err, i, j)
        
                # Buffer grid cell
                sa_sf = inv_std.sa.copy()
                _, p = inv_std.estimate_p(pert)
                sa_sf[0] *= p**2
                inv_buff = inv.Inversion(
                    rs=j, 
                    xa_abs=xa_abs, 
                    sa=sa_sf, 
                    nobs_per_cell=nobs,
                    U=U,
                    gamma=1, 
                    BC=true_BC.BCt + pert
                )
                pert_base(inv_buff, inv_std, 'buffer', 'prior', err, i, j)

    # So variations
    # print('  So')
    for i, nobs in enumerate(nobs_per_cell):
        # Standard inversion
        var_so = inv.Inversion(
            rs=j, nobs_per_cell=nobs, gamma=1, BC=true_BC.BCt, U=wind)
        # prev, R = var_so.preview_2d(pert)
        # print(nobs, 1/R)

        # Perturb the boundary condition and calculate the error
        var_so_pert = inv.Inversion(
            rs=j, nobs_per_cell=nobs, gamma=1, BC=true_BC.BCt + pert, U=wind)
        pert_standard(var_so_pert, var_so, pert, 'nobs', err, i, j)

        # BC correction
        var_so_opt_BC = inv.Inversion(
            rs=j, nobs_per_cell=nobs, gamma=1, BC=true_BC.BCt + pert, U=wind,
            opt_BC=True)
        pert_base(var_so_opt_BC, var_so, 'boundary', 'nobs', err, i, j)

        # Buffer grid cell
        sa_sf = var_so.sa.copy()
        _, p = var_so.estimate_p(pert)
        sa_sf[0] *= p**2
        var_so_buff = inv.Inversion(
            rs=j, sa=sa_sf, nobs_per_cell=nobs, gamma=1, BC=true_BC.BCt + pert, U=wind)
        pert_base(var_so_buff, var_so, 'buffer', 'nobs', err, i, j)

    # print('  U')
    for i, U in enumerate(Us):
        U = np.concatenate([np.arange(U + 2, U - 2, -1), 
                            np.arange(U - 2, U + 2, 1)])
        U *= 24*60*60/1000 # convert from m/s to km/day

        # Standard inversion
        var_U = inv.Inversion(rs=j, U=U, gamma=1, BC=true_BC.BCt)

        # Perturb the boundary condition and calculate the error
        var_U_pert = inv.Inversion(
            rs=j, U=U, gamma=1, BC=true_BC.BCt + pert)
        pert_standard(var_U_pert, var_U, pert, 'U', err, i, j)

        # BC correction
        var_U_opt_BC = inv.Inversion(
            rs=j, U=U, gamma=1, BC=true_BC.BCt + pert, opt_BC=True)
        pert_base(var_U_opt_BC, var_U, 'boundary', 'U', err, i, j)

        # Buffer grid cell
        sa_sf = var_U.sa.copy()
        _, p = var_U.estimate_p(pert)
        sa_sf[0] *= p**2
        var_U_buff = inv.Inversion(
            rs=j, sa=sa_sf, U=U, gamma=1, BC=true_BC.BCt + pert)
        pert_base(var_U_buff, var_U, 'buffer', 'U', err, i, j)

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
        #   fp.color(1, 'plasma'), # 1D preview
          fp.color(4, 'plasma'), # 2D preview
          fp.color(7, 'plasma')] # diagnostic]
labels = ['True error', 'Boundary method',
          'Buffer method', 'Preview', 'Diagnostic']
for k, var in enumerate(['_ils']):
    for j, intervention in enumerate(['standard', 'boundary', 'buffer']):
        # Combine the sa and so modifications
        print(intervention)
        data = err[intervention]
        data['all'] = {}
        for quant, _ in data['prior'].items():
            data['all'][quant] = np.concatenate(
                [data['nobs'][quant], 
                 data['prior'][quant]], axis=0)

        # Reduce R by the number of obs per grid cell so that it better
        # reflects the actual observational uncertainty
        x_r = 1/err['standard']['all']['R'].mean(axis=1)
        idx = np.argsort(x_r)
        x_r = x_r[idx]#/config['nobs_per_cell']
        y_r = data['all'][f'err{var}']
        ax[2*k].plot(x_r, y_r.mean(axis=1)[idx],
                   color=colors[j], ls=ls[j], lw=lw[j],
                   label=labels[j])
        ax[2*k].fill_between(
            x_r,
            y_r.mean(axis=1)[idx] - y_r.std(axis=1)[idx],
            y_r.mean(axis=1)[idx] + y_r.std(axis=1)[idx], 
            color=colors[j], alpha=0.25)

        # Wind speed
        x_u = pert/(true_BC.L/(xs['U']*24*60*60/1000))
        y_u = data['U'][f'err{var}']
        ax[2*k + 1].plot(
            x_u, y_u.mean(axis=1), color=colors[j], ls=ls[j], lw=lw[j])
        ax[2*k + 1].fill_between(
            x_u,
            y_u.mean(axis=1) - y_u.std(axis=1),
            y_u.mean(axis=1) + y_u.std(axis=1), 
            color=colors[j], alpha=0.25)

        # Diagnostics
        if intervention == 'standard':
            for i, quant in enumerate(['prev']):#, 'diag']):
                y_r = data['all'][f'{quant}{var}']
                ax[2*k].plot(
                    x_r, 
                    y_r.mean(axis=1)[idx], 
                    color=colors[i + 3], ls=ls[i + 3], lw=lw[i + 3],
                    label=labels[i + 3])
                ax[2*k].fill_between(
                    x_r,
                    y_r.mean(axis=1)[idx] - y_r.std(axis=1)[idx],
                    y_r.mean(axis=1)[idx] + y_r.std(axis=1)[idx], 
                    color=colors[i + 3], alpha=0.25)
                y_u = data['U'][f'{quant}{var}']
                ax[2*k + 1].plot(
                    x_u, 
                    y_u.mean(axis=1), 
                    color=colors[i + 3], ls=ls[i + 3], lw=lw[i + 3])
                ax[2*k + 1].fill_between(
                    x_u,
                    y_u.mean(axis=1) - y_u.std(axis=1),
                    y_u.mean(axis=1) + y_u.std(axis=1), 
                    color=colors[i + 3], alpha=0.25)
                
ax[0].set_xlim(0, 15)
ax[1].set_xlim(100, 300)
ax[0].set_ylim(0, 12)
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
reorder = [-1, 0, 1,
           -1, 2, 3]
        #    -1, 3, 4]

handles = [handles[i] for i in reorder]
labels = [labels[i] for i in reorder]
# labels[3] = 'Error estimates : '
labels[3] = 'Correction methods : '

ax[0].legend(handles=handles, labels=labels,
             loc='upper center', alignment='center',  
             bbox_to_anchor=(0.5, -0.2), bbox_transform=fig.transFigure,
             ncol=2, handlelength=2, frameon=False, 
             fontsize=ps.LABEL_FONTSIZE*ps.SCALE) 

ax[0] = fp.add_title(ax[0], 'Varying error ratio')
ax[0] = fp.add_labels(
    ax[0], r'Information ratio $R$ [unitless]', 
    'Influence length scale [unitless]')
    # 'First grid cell index where\n'r'$\Delta \hat{x}/x_A \leq 0.25$ [unitless]')
ax[1] = fp.add_title(ax[1], 'Varying residence time')
ax[1] = fp.add_labels(
    ax[1], 
    'Background 'r'uncertainty $\tau_k^{-1}\Delta c$ [ppb day$^{-1}$]', '')

ax[0].text(0.025, 0.975, '(a)', fontsize=ps.LABEL_FONTSIZE*ps.SCALE,
           transform=ax[0].transAxes, ha='left', va='top')
ax[1].text(0.025, 0.975, '(b)', fontsize=ps.LABEL_FONTSIZE*ps.SCALE,
           transform=ax[1].transAxes, ha='left', va='top')

ax[0].set_xscale('log')
ax[0].set_xlim(0.03, 15)

# ax[2].set_ylim(0, 5)
# ax[3].set_ylim(0, 5)
# fp.save_fig(fig, plot_dir, f'parameter_sensitivity')
# %%
