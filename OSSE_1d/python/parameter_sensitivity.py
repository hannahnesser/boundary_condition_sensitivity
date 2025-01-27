#%%
import numpy as np
import os
from copy import deepcopy as dc
from matplotlib import rcParams
from matplotlib.patches import Patch as patch

# Custom packages
import settings as s
from utilities import plot_settings as ps
from utilities import inversion as inv
from utilities import format_plots as fp

# rcParams['text.usetex'] = True
np.set_printoptions(precision=2, linewidth=300, suppress=True)

## -------------------------------------------------------------------------##
# File Locations
## -------------------------------------------------------------------------##
plot_dir = f'../plots/n{s.nstate}_m{s.nobs}'
# plot_dir = f'{plot_dir}/n{s.nstate}_m{s.nobs}'
if not os.path.exists(plot_dir):
    os.mkdir(plot_dir)

## -------------------------------------------------------------------------##
# Get the true inversion
## -------------------------------------------------------------------------##
U = np.concatenate([np.arange(5, 0, -1), 
                    np.arange(1, 5, 1)])*24
U = np.repeat(U, 2)
# U = 5*24

if type(U) in [float, int]:
    suffix = 'constwind'
else:
    suffix = 'varwind'

true_BC = inv.Inversion(gamma=1, U=U)
true_BC_opt = inv.Inversion(gamma=1, opt_BC=True, U=U)

## -------------------------------------------------------------------------##
# Define mean error tests
## -------------------------------------------------------------------------##
def get_tot_err(xhat, truth):
    return ((xhat - truth)**2).sum()**0.5

def get_half_err(xhat, truth):
    # nhalf = int(len(xhat)/2)
    return ((xhat[1:] - truth[1:])**2).sum()**0.5

## -------------------------------------------------------------------------##
## Varying inversion parameters
## -------------------------------------------------------------------------##
#%%
nsamples = int(50) 
sas = np.arange(0.25, 2.25, 0.25)
sos = np.arange(5, 45, 5)
Us = 24*np.array([2.5, 5, 7.5, 10, 12.5, 15])
var_params = {'sa' : np.zeros((len(sas), nsamples)), 
              'so' : np.zeros((len(sos), nsamples)), 
              'U' : np.zeros((len(Us), nsamples))}
tot_err = {'standard' : dc(var_params), 'correction' : dc(var_params), 
           'buffer' : dc(var_params), 'combination' : dc(var_params),
           'prior' : dc(var_params)}
tot_err = {'constwind' : dc(tot_err), 'varwind' : dc(tot_err)}
        #    'sequential' : var_params.copy()}
half_err = dc(tot_err)
# xa1s = []

# In all cases, use a 10 ppb perturbation
pert = 10

# Run it for both 
for suffix in ['constwind', 'varwind']:
    for j in range(nsamples):
        if j % 10 == 0:
            print(j, '/', nsamples)
        j = int(j)
        # Sa variations
        print('  Sa')
        for i, sa in enumerate(sas):
            sa = float(sa)

            # Standard inversion
            var_sa = inv.Inversion(rs=j, sa=sa, gamma=1, BC=true_BC.BCt + pert, U=U)
            tot_err[suffix]['standard']['sa'][i, j] = get_tot_err(
                var_sa.xhat*var_sa.xa_abs, var_sa.xt_abs)
            half_err[suffix]['standard']['sa'][i, j] = get_half_err(
                var_sa.xhat*var_sa.xa_abs, var_sa.xt_abs)
            
            # Get the prior in the first grid cell
            # xa1s.append(var_sa.xa_abs[0])

            # BC correction
            var_sa_opt_BC = inv.Inversion(rs=j, sa=sa, gamma=1, 
                                        BC=true_BC.BCt + pert, opt_BC=True, U=U)
            tot_err[suffix]['correction']['sa'][i, j] = get_tot_err(
                var_sa_opt_BC.xhat*var_sa_opt_BC.xa_abs, 
                            var_sa_opt_BC.xt_abs)
            half_err[suffix]['correction']['sa'][i, j] = get_half_err(
                var_sa_opt_BC.xhat*var_sa_opt_BC.xa_abs,
                            var_sa_opt_BC.xt_abs)
            
            # Buffer grid cell
            sa_sf = var_sa.sa.copy()
            sa_sf[0] *= 10**2
            var_sa_buff = inv.Inversion(rs=j, sa=sa_sf, gamma=1, 
                                        BC=true_BC.BCt + pert, U=U)
            tot_err[suffix]['buffer']['sa'][i, j] = get_tot_err(
                var_sa_buff.xhat*var_sa_buff.xa_abs, var_sa_buff.xt_abs)
            half_err[suffix]['buffer']['sa'][i, j] = get_half_err(
                var_sa_buff.xhat*var_sa_buff.xa_abs, var_sa_buff.xt_abs)
            
            # Combination
            var_sa_combo = inv.Inversion(rs=j, sa=sa_sf, gamma=1, 
                                        BC=true_BC.BCt + pert, opt_BC=True, U=U)
            tot_err[suffix]['combination']['sa'][i, j] = get_tot_err(
                var_sa_combo.xhat*var_sa_combo.xa_abs, var_sa_combo.xt_abs)
            half_err[suffix]['combination']['sa'][i, j] = get_half_err(
                var_sa_combo.xhat*var_sa_combo.xa_abs, 
                            var_sa_combo.xt_abs)
            
            # Prior
            tot_err[suffix]['prior']['sa'][i, j] = get_tot_err(
                var_sa.xa_abs, var_sa.xt_abs)
            half_err[suffix]['prior']['sa'][i, j] = get_half_err(
                var_sa.xa_abs, var_sa.xt_abs)

        # So variations
        print('  So')
        for i, so in enumerate(sos):
            so = float(so)

            # Standard inversion
            var_so = inv.Inversion(
                rs=j, so=so, gamma=1, BC=true_BC.BCt + pert, U=U)
            tot_err[suffix]['standard']['so'][i, j] = get_tot_err(
                var_so.xhat*var_so.xa_abs, var_so.xt_abs)
            half_err[suffix]['standard']['so'][i, j] = get_half_err(
                var_so.xhat*var_so.xa_abs, var_so.xt_abs)

            # BC correction
            var_so_opt_BC = inv.Inversion(rs=j, so=so, gamma=1, 
                                        BC=true_BC.BCt + pert, opt_BC=True, U=U)
            tot_err[suffix]['correction']['so'][i, j] = get_tot_err(
                var_so_opt_BC.xhat*var_so_opt_BC.xa_abs, 
                            var_so_opt_BC.xt_abs)
            half_err[suffix]['correction']['so'][i, j] = get_half_err(
                var_so_opt_BC.xhat*var_so_opt_BC.xa_abs,
                            var_so_opt_BC.xt_abs)
            
            # Buffer grid cell
            sa_sf = true_BC.sa.copy()
            sa_sf[0] *= 10**2
            var_so_buff = inv.Inversion(rs=j, so=so, sa=sa_sf, gamma=1, 
                                        BC=true_BC.BCt + pert, U=U)
            tot_err[suffix]['buffer']['so'][i, j] = get_tot_err(
                var_so_buff.xhat*var_so_buff.xa_abs, var_so_buff.xt_abs)
            half_err[suffix]['buffer']['so'][i, j] = get_half_err(
                var_so_buff.xhat*var_so_buff.xa_abs, var_so_buff.xt_abs)
            
            # Combination
            var_so_combo = inv.Inversion(rs=j, so=so, sa=sa_sf, gamma=1, 
                                        BC=true_BC.BCt + pert,
                                        opt_BC=True, U=U)
            tot_err[suffix]['combination']['so'][i, j] = get_tot_err(
                var_so_combo.xhat*var_so_combo.xa_abs, var_so_combo.xt_abs)
            half_err[suffix]['combination']['so'][i, j] = get_half_err(
                var_so_combo.xhat*var_so_combo.xa_abs, var_so_combo.xt_abs)
            
            # Prior
            tot_err[suffix]['prior']['so'][i, j] = get_tot_err(var_so.xa_abs, var_so.xt_abs)
            half_err[suffix]['prior']['so'][i, j] = get_half_err(var_so.xa_abs, 
                                                        var_so.xt_abs)        
        
        print('  U')
        for i, U in enumerate(Us):
            if suffix == 'varwind':
                U = np.concatenate([np.arange(U, 0, -1), 
                                    np.arange(1, U, 1)])*24
                U = np.repeat(U, 2)

            # Standard inversion
            var_U = inv.Inversion(rs=j, U=U, gamma=1, BC=true_BC.BCt + pert)
            tot_err[suffix]['standard']['U'][i, j] = get_tot_err(
                var_U.xhat*var_U.xa_abs, var_U.xt_abs)
            half_err[suffix]['standard']['U'][i, j] = get_half_err(
                var_U.xhat*var_U.xa_abs, var_U.xt_abs)

            # BC correction
            var_U_opt_BC = inv.Inversion(rs=j, U=U, gamma=1, BC=true_BC.BCt + pert, 
                                        opt_BC=True)
            tot_err[suffix]['correction']['U'][i, j] = get_tot_err(
                var_U_opt_BC.xhat*var_U_opt_BC.xa_abs, var_U_opt_BC.xt_abs)
            half_err[suffix]['correction']['U'][i, j] = get_half_err(
                var_U_opt_BC.xhat*var_U_opt_BC.xa_abs, var_U_opt_BC.xt_abs)
            
            # Buffer grid cell
            sa_sf = true_BC.sa.copy()
            sa_sf[0] *= 10**2
            var_U_buff = inv.Inversion(rs=j, U=U, sa=sa_sf, gamma=1, 
                                    BC=true_BC.BCt + pert)
            tot_err[suffix]['buffer']['U'][i, j] = get_tot_err(
                var_U_buff.xhat*var_U_buff.xa_abs, var_U_buff.xt_abs)
            half_err[suffix]['buffer']['U'][i, j] = get_half_err(
                var_U_buff.xhat*var_U_buff.xa_abs, var_U_buff.xt_abs)
            
            # Combination
            var_U_combo = inv.Inversion(rs=j, U=U, sa=sa_sf, gamma=1, 
                                        BC=true_BC.BCt + pert,
                                        opt_BC=True)
            tot_err[suffix]['combination']['U'][i, j] = get_tot_err(
                var_U_combo.xhat*var_U_combo.xa_abs, var_U_combo.xt_abs)
            half_err[suffix]['combination']['U'][i, j] = get_half_err(
                var_U_combo.xhat*var_U_combo.xa_abs, var_U_combo.xt_abs)
            
            # Prior
            tot_err[suffix]['prior']['U'][i, j] = get_tot_err(var_U.xa_abs, var_U.xt_abs)
            half_err[suffix]['prior']['U'][i, j] = get_half_err(var_U.xa_abs, 
                                                        var_U.xt_abs)

#%%
## -------------------------------------------------------------------------##
## Plot output
## -------------------------------------------------------------------------##
fig, ax = fp.get_figax(rows=1, cols=3, aspect=1, 
                        max_height=4.5, max_width=6,
                        sharex=False, sharey=True)
                    #    sharex=True, sharey=True)
# ax = ax.T.flatten()
# plt.subplots_adjust(wspace=0.5, hspace=0.5)
# ax[0].set_ylim(-1.05, 1.05)

# fp.save_fig(fig, plot_dir, f'constant_BC_{suffix}')
# var_params = {'sa' : [], 'so' : [], 'U' : []}
# tot_err = {'standard' : var_params.copy(), 'correction' : var_params.copy(), 
#            'buffer' : var_params.copy(), 'combined' : var_params.copy()}
xs = {'sa' : sas, 'so' : sos, 'U' : Us/24}
ls = ['-', '--']
hatch = ['///', '\\\\\\']
for k, wind in enumerate(['constwind', 'varwind']):
    for j, intervention in enumerate(['standard', 'correction', 
                                    'buffer']):# 'combination']):
        if intervention in ['standard', 'correction']:
            data = tot_err[wind][intervention]
        else:
            data = half_err[wind][intervention]
        for i, param in enumerate(['sa', 'so', 'U']):
            if param == 'so':
                delta = 0.1
            else:
                delta = 0.01

            # ax[i].plot(xs[param] + delta*j, data[param].mean(axis=1), 
            #         ls=ls[k], color=fp.color(j*2 + 1, lut=10))
            # ax[i].plot(
            #     xs[param] + delta*j, 
            #     data[param].mean(axis=1) - data[param].std(axis=1),
            #     color=fp.color(j*2 + 1, lut=10), alpha=0.5, ls='-', lw=0.2)
            # ax[i].plot(
            #     xs[param] + delta*j, 
            #     data[param].mean(axis=1) + data[param].std(axis=1), 
            #     color=fp.color(j*2 + 1, lut=10), alpha=0.5, ls='-', lw=0.2)
            ax[i].plot(xs[param], data[param].mean(axis=1), 
                       ls=ls[k], color=fp.color(j*3 + 1, lut=10))
            ax[i].fill_between(
                xs[param],
                data[param].mean(axis=1) - data[param].std(axis=1),
                data[param].mean(axis=1) + data[param].std(axis=1), 
                color=fp.color(j*3 + 1, lut=10), hatch=hatch[k], alpha=0.25)

            ax[i].plot(xs[param], tot_err[suffix]['prior'][param].mean(axis=1), 
                    ls='--', color='grey')
            ax[i].fill_between(
                xs[param],
                tot_err[suffix]['prior'][param].mean(axis=1) - 
                tot_err[suffix]['prior'][param].std(axis=1),
                tot_err[suffix]['prior'][param].mean(axis=1) + 
                tot_err[suffix]['prior'][param].std(axis=1), 
                color='grey', alpha=0.05)
            # ax[i].plot(xs[param], half_err[suffix]['prior'][param].mean(axis=1), 
            #            ls='--', lw=2, color='black')
            # ax[i].fill_between(
            #     xs[param],
            #     half_err[suffix]['prior'][param].mean(axis=1) - 
            #     half_err[suffix]['prior'][param].std(axis=1),
            #     half_err[suffix]['prior'][param].mean(axis=1) + 
            #     half_err[suffix]['prior'][param].std(axis=1), 
            #     color='grey', alpha=0.05)
            ax[i].set_xlim(xs[param].min(), xs[param].max())

    # # Plot approximate threshold
    # xb = true_BC_opt.xa[-1]
    # sb = true_BC_opt.sa[-1]**0.5/xb
    # xa = true_BC_opt.xa[0]*np.array(xa1s).mean()
    # sa = true_BC_opt.sa[0]
    # tau = true_BC_opt.tau
    # p = 10

    # idx = np.where(xb*sb/((p**2 - 1)**0.5*xs['sa']*xa*tau) > 1)[0][-1]
    # ax[0].axvline((xs['sa'][idx] + xs['sa'][idx + 1])/2, ls=':', lw=2, color='grey',
    #               label='alpha')

    # rat = xb*sb/((p**2 - 1)**0.5*sa*xa*tau)
    # if rat < 1:
    #     print('ax[1] should always have sigma_b / sigma_a > 1')

    # idx = np.where(xb*sb/((p**2 - 1)**0.5*sa*xa*xs['U']) > 1)[0]
    # try:
    #     ax[0].axvline((xs['tau'][idx[-1]] + xs['sa'][idx[-1] + 1])/2, ls=':', 
    #                   lw=2, color='grey')
    # except:
    #     print('No values satisfy sigma_b / sigma_a > 1')

# Custom legend
custom_patches = [patch(facecolor=fp.color(1, lut=10), alpha=0.25),
                patch(facecolor=fp.color(4, lut=10), alpha=0.25),
                patch(facecolor=fp.color(7, lut=10), alpha=0.25), 
                # patch(facecolor=fp.color(7, lut=10), alpha=0.2),
                patch(facecolor='grey', alpha=0.25),
                patch(facecolor='grey', alpha=0.25, hatch='///'),
                patch(facecolor='grey', alpha=0.25, hatch='\\\\\\')]
custom_labels = ['Standard inversion', 'Correction method',
                'Buffer method', 'Prior - true emissions\nRMSE',
                'Constant wind speeds', 'Varying wind speeds']
ax[0].legend(handles=custom_patches, labels=custom_labels,
            bbox_to_anchor=(0.5, -0.2), loc='center', ncol=3,
            bbox_transform=fig.transFigure, frameon=False,
            fontsize=ps.LABEL_FONTSIZE*ps.SCALE)

ax[0] = fp.add_title(ax[0], 'Varying prior error')
ax[0] = fp.add_labels(ax[0], 'Prior error standard\ndeviation [unitless]', 
                    'Root mean square error\n(RMSE) [ppb/day]')
ax[1] = fp.add_title(ax[1], 'Varying observing\nsystem error')
ax[1] = fp.add_labels(ax[1], 'Observing system error\nstandard deviation [ppb]', '')
ax[2] = fp.add_title(ax[2], 'Varying lifetime')
ax[2] = fp.add_labels(ax[2], 'Maximum wind speed [km/hr]', '')

fp.save_fig(fig, plot_dir, f'parameter_sensitivity_{suffix}')
    # %%
