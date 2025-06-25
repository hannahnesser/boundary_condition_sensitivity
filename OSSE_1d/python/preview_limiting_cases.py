#%%
import numpy as np
import os
from copy import deepcopy as dc
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Custom packages
from utilities import plot_settings as ps
from utilities import utils
from utilities import inversion as inv
from utilities import format_plots as fp

# rcParams['text.usetex'] = True
np.set_printoptions(precision=2, linewidth=300, suppress=True)

## -------------------------------------------------------------------------##
# File Locations
## -------------------------------------------------------------------------##
project_dir, config = utils.setup()
plot_dir = f'{project_dir}/plots'
if not os.path.exists(plot_dir):
    os.mkdir(plot_dir)

# def estimate_delta_xhat(U=5*24, sa_i=None, sa_up=None, so_i=None, so_up=None,
#                         idx=2):
#     true_BC = inv.Inversion(U=U, gamma=1)
#     L = true_BC.L
#     nstate = true_BC.nstate
#     D = np.arange(L/2, L*nstate + L/2, L)[idx]
#     xa_abs = true_BC.xt_abs
#     x_up = np.append(0, np.cumsum(xa_abs))[:-1] + xa_abs/2
#     x_up = x_up[idx]
#     if U is None:
#         U = np.abs(true_BC.U).mean()
#     if sa_i is None:
#         sa_i = 0.25
#         print('Base value sa_i : ', sa_i)
#     if sa_up is None:
#         sa_up = 0.25
#         print('Base value sa_up : ', sa_up)
#     if so_i is None:
#         so_i = 10**2 #((true_BC.so.mean()/true_BC.nobs_per_cell)**0.5)**2
#         print('Base value so_i : ', so_i)
#     if so_up is None:
#         so_up = 5**2 #((true_BC.so.mean()/true_BC.nobs_per_cell)**0.5)**2 * L/D
#         print('Base value so_up : ', so_up)

#     k_i = L/U
#     k_up = D/U

#     kx_i = k_i * xa_abs[idx]
#     kx_up = k_up * x_up
#     sa_up_so_up = sa_up / so_up
#     so_up_sa_i = so_up / sa_i
#     so_i_sa_i = so_i / sa_i

#     num = kx_i # Normalize by the perturbation
#     den1 = kx_up**2 * sa_up_so_up * ( kx_i**2 + so_i_sa_i + so_up_sa_i ) 
#     den2 = kx_i**2
#     den3 = so_i_sa_i
#     delta_xhat = - num / (den1 + den2 + den3)

#     return float(delta_xhat)

# #%%
# Us = np.arange(1, 10)
# U_eff = []
# for U in Us:
#     U_eff.append(estimate_delta_xhat(U=U*24))
# #%%

# get_info = estimate_delta_xhat()
# #%%
# sas = np.arange(0.25, 2.1, 0.25)**2
# sos = np.arange(1, 11, 1)**2
# # sos = np.sort(np.insert(sos, 0, 4/3))
# eff = np.zeros((len(sas), len(sas), len(sos), len(sos)))
# print(eff.shape)
# print('Iterating through', len(sas)*len(sas)*len(sos)*len(sos), 'elements')
# for i, sa_i in enumerate(sas):
#     for j, sa_up in enumerate(sas):
#         for k, so_i in enumerate(sos):
#             for l, so_up in enumerate(sos):
#                 eff[i, j, k, l] = estimate_delta_xhat(
#                     sa_i=sa_i, sa_up=sa_up, so_i=so_i, so_up=so_up
#                 )
# # %%
# sa_i_base_idx = np.argwhere(sas == 0.25)[0][0]
# sa_up_base_idx = np.argwhere(sas == 0.25)[0][0]
# so_i_base_idx = np.argwhere(sos == 100)[0][0]
# so_up_base_idx = np.argwhere(sos == 25)[0][0]

# eff_sa_up_so_up = eff[sa_i_base_idx, :, so_i_base_idx, :]
# eff_sa_i_so_i = eff[:, sa_up_base_idx, :, so_up_base_idx]
# eff_so_i_so_up = eff[sa_i_base_idx, sa_up_base_idx, :, :]
# eff_sa_i_sa_up = eff[:, :, so_i_base_idx, so_up_base_idx]
# eff_sa_up_so_i = eff[sa_i_base_idx, :, :, so_up_base_idx]
# eff_sa_i_so_up = eff[:, sa_up_base_idx, so_i_base_idx, :]

# all_effs = -99*np.ones(
#     (2*len(sos) + len(sas) + 2, 
#      2*len(sas) + len(sos) + 2))
# all_effs[:len(sos), :len(sas)] = eff_sa_up_so_i.T
# all_effs[len(sos) + 1:2*len(sos) + 1, :len(sas)] = eff_sa_up_so_up.T
# all_effs[2*len(sos) + 2:, :len(sas)] = eff_sa_i_sa_up
# all_effs[:len(sos), len(sas) + 1:2*len(sas) + 1] = eff_sa_i_so_i.T
# all_effs[:len(sos), 2*len(sas) + 2:] = eff_so_i_so_up
# all_effs[len(sos) + 1:2*len(sos) + 1, len(sas) + 1:2*len(sas) + 1] = eff_sa_i_so_up.T

# all_effs[all_effs == -99] = np.nan

# #%%
# fig, ax = fp.get_figax(aspect=1, max_width=7.5)
# c = ax.imshow(100*all_effs, cmap=fp.cmap_trans('viridis', reverse=True),
#               vmin=-2, vmax=0)
# cax = fp.add_cax(fig, ax)
# cb = fig.colorbar(c, ax=ax, cax=cax)
# cb = fp.format_cbar(cb, cbar_title=r'Preview value [% ppb$^{-1}$]')

# ylabels = np.concatenate(
#     [(sos**0.5).astype(int), 
#      np.array([None]), 
#      (sos**0.5).astype(int), 
#      np.array([None]), 
#      (100*sas**0.5).astype(int)], 
#      axis=0)
# xlabels = np.concatenate(
#     [(100*sas**0.5).astype(int), 
#      np.array([None]), 
#      (100*sas**0.5).astype(int), 
#      np.array([None]), 
#      (sos**0.5).astype(int)], 
#      axis=0)

# ax.set_yticks(np.arange(all_effs.shape[0]), ylabels, 
#               fontsize=ps.TICK_FONTSIZE*ps.SCALE)
# ax.set_xticks(np.arange(all_effs.shape[1]), xlabels, 
#               fontsize=ps.TICK_FONTSIZE*ps.SCALE)
# ax.xaxis.tick_top()
# ax.tick_params(axis='x', rotation=90)

# ax.text(0.5*len(sas) - 0.5, -5, 'Upstream\nprior error [%]', 
#         ha='center', va='center', fontsize=ps.LABEL_FONTSIZE*ps.SCALE)
# ax.text(1.5*len(sas) + 0.5, -5, 'Grid cell\nprior error [%]', 
#         ha='center', va='center', fontsize=ps.LABEL_FONTSIZE*ps.SCALE)
# ax.text(2*len(sas) + 0.5*len(sos) + 1.5, -5, 
#         'Upstream\nobserving\nsystem\nerror [ppb]', 
#         ha='center', va='center', fontsize=ps.LABEL_FONTSIZE*ps.SCALE)

# ax.text(-6, 0.5*len(sos) - 0.5, 'Grid cell\nobserving\nsystem\nerror [ppb]',
#         fontsize=ps.LABEL_FONTSIZE*ps.SCALE, rotation=90, 
#         va='center', ha='center')
# ax.text(-6, 1.5*len(sos) + 0.5, 'Upstream\nobserving\nsystem\nerror [ppb]',
#         fontsize=ps.LABEL_FONTSIZE*ps.SCALE, rotation=90, 
#         va='center', ha='center')
# ax.text(-6, 2*len(sos) + 0.5*len(sas) + 1.5, 
#         'Grid cell\nprior error [%]',
#         fontsize=ps.LABEL_FONTSIZE*ps.SCALE, rotation=90, 
#         va='center', ha='center')
# # # ax.text(1.5*len(sas) + 1, -3, 'Grid cell\nprior error [%]', ha='center',
# # #         fontsize=ps.LABEL_FONTSIZE*ps.SCALE)
# # # ax.text(2.5*len(sas) + 2, -3, 'Upstream\nobserving\nsystem\nerror [ppb]', 
# # #         ha='center', fontsize=ps.LABEL_FONTSIZE*ps.SCALE)

# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# ax.spines['bottom'].set_visible(False)
# ax.spines['left'].set_visible(False)

# fp.save_fig(fig, plot_dir, 'limiting_analysis_1')
# # %%


# fig, ax = fp.get_figax(aspect=1.25, cols=3)
# ax[0].scatter((100*sas[:, None]**0.5/sos[None, :]**0.5),
#               eff_sa_up_so_up)
# ax[1].scatter((100*sas[:, None]**0.5/sos[None, :]**0.5),
#               eff_sa_i_so_up)
# ax[2].scatter((100*sas[:, None]**0.5/sos[None, :]**0.5),
#               eff_sa_i_so_i)
#%%

from matplotlib import cm
n = 10
def preview_1d(sa_bc, k, sa, so, n=n):
    # Define secondary variables
    # R = (so/(k*sa))**2
    k = k*np.tril(np.ones((n, n))) # ppb/flux
    so_inv = np.diag(1/(so**2*np.ones(n))) # 1/ppb2
    sa_inv = np.diag(1/(sa**2*np.ones(n))) # 1/flux2
    g = np.linalg.inv(sa_inv + k.T @ so_inv @ k) @ k.T @ so_inv
    return -g.sum(axis=1) # Normalized by delta_c

def R_gt_one(sa_bc, k, R=1e-10, n=n):
    j = np.arange(1, n + 1)
    return -(1/k)*R**(1 - j) # Normalized by delta_c

def R_lt_one(sa_bc, k, R=1e10, n=n):
    j = np.arange(1, n + 1)
    return -(1/k)*R*(n - j + 1) # Normalized by delta_c

true_BC = inv.Inversion(gamma=1, nstate=n)
tau_k = true_BC.tau # day
delta_c = 1
xa = 25
sa = (xa*0.5)**2 # ppb/day
so = 10**2 # ppb
R_base = (tau_k**2*sa)/so
print(R_base)

fig, ax = fp.get_figax(rows=2, aspect=1.5, max_width=4, max_height=4, 
                       sharey=True, sharex=False, height_ratios=[0.5, 1])
xs = np.arange(1, n + 1)
fp.add_title(ax[0], 'BC-induced posterior error for\ninversions with constant parameters',
             y=1.1)

R_lt = R_lt_one(delta_c, k=tau_k, R=1*R_base)
R_lt_sum = 100*np.sum(R_lt)/(xa*n)
ax[0].plot(xs + 0.5, R_lt/xa*100, color=fp.color(3, cmap='Grays', lut=7), lw=2,
           ls=':', label=r'R $\ll$ 1')#, $Total error$ = $'f'{R_lt_sum:.2f} %)')
# ax[0].plot(xs + 0.5, preview_1d(delta_c, tau_k, sa, so/1, n)/xa*100, 
#            color=fp.color(3, cmap='Greens', lut=7), lw=2, ls='--',
#            label=r'Preview')#, $Total error$ = $'f'{R_lt_sum:.2f} %)')

n_gt = 5e4
R_gt = R_gt_one(delta_c, k=tau_k, R=n_gt*R_base)
R_gt_sum = 100*np.sum(R_gt)/(xa*n)
ax[0].plot(xs + 0.5, R_gt/xa*100, color=fp.color(5, cmap='Grays', lut=7), lw=2,
           ls=':', label=r'R $\gg$ 1')#, $Total error$ = $'f'{R_gt_sum:.2f} %)')
# ax[0].plot(xs + 0.5, preview_1d(delta_c, tau_k, sa, so/n_gt, n)/xa*100, 
#            color=fp.color(5, cmap='Greens', lut=7), lw=2, ls='--',
#            label=r'Preview')#, $Total error$ = $'f'{R_lt_sum:.2f} %)')

ax[0].set_ylim(-25, 2.5)

ax[0].set_xticks(xs + 0.5)
ax[0].set_xticklabels(xs, fontsize=ps.TICK_FONTSIZE*ps.SCALE)
ax[0].set_xlim(1, n + 1)
for j in range(21):
    ax[0].axvline(j + 1, c=fp.color(1, lut=11), alpha=0.2, ls=':', lw=0.5)

fp.add_labels(ax[0], 'Grid cells from upwind boundary', 
              'BC-induced error\n'r'[% ppb$^{-1}$]')

Rs = []
prevs = []
tot_err = []
for sasa in np.arange(0.1, 20, 0.25): # Assuming xa = xa
    for soso in np.arange(1, 21, 2):
        Rs.append(1/((soso/(tau_k*sasa*25))**2))
        prev_i = preview_1d(delta_c, tau_k, sasa*25, soso)
        prevs.append(prev_i/xa*100)
        tot_err.append(100*(np.sum(prev_i)/(xa*n)))

x = np.array(Rs)
idx = np.argsort(x)
for i in range(n):
    y = np.array([p[i] for p in prevs])
    if i == 5:
        label = 'Grid cell error'
    else:
        label = None
    ax[1].plot(x[idx], y[idx], color=fp.color(i, lut=n + 1, cmap='plasma'),
               lw=0.5, label=label)

# ax1 = ax[1].twinx()
# ax1.plot(x[idx], np.array(tot_err)[idx], 
#          color='lightgrey', lw=2, ls='-', label='Total error')
ax[1].plot(x[idx], np.array(tot_err)[idx], 
         color='black', lw=2, ls='-', label='Total error')
# ax1.set_ylim(-8, 0.8)
# point1 = np.array(tot_err)[idx][np.argmin(np.abs(x - 1*R_base))]
# point2 = np.array(tot_err)[idx][np.argmin(np.abs(x - n_gt*R_base))]
# ax1.scatter(1*R_base, R_lt_sum, 
#             color=fp.color(3, cmap='Grays', lut=7), s=50, marker='o')
# ax1.scatter(n_gt*R_base, R_gt_sum, 
#             color=fp.color(5, cmap='Grays', lut=7), s=50, marker='o') 
# fp.add_labels(ax1, '', 'Total error [% ppb$^{-1}$]')

import matplotlib as mpl
cmap = mpl.colormaps['plasma'].resampled(n + 1)
cmap = mpl.colors.ListedColormap(cmap(np.linspace(0, 1, n + 1)))
bounds = np.arange(1, n + 2)
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
cax = fp.add_cax(fig, ax[1], cbar_pad_inches=1.3, horizontal=True)
cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax,
                  ticks=((bounds[:-1] + bounds[1:])/2), 
                  orientation='horizontal')
cb.ax.set_xticklabels([f'{m:d}' for m in range(1, n + 1)])
cb.ax.minorticks_off()
fp.format_cbar(cb, cbar_title='Grid cells from\nupwind boundary',
               horizontal=True)

ax[1].set_xscale('log')
ax[1].set_xlim(0.33*R_base, (n_gt*3)*R_base)
ax[1].axvline(1*R_base, lw=2, color=fp.color(3, cmap='Grays', lut=7), ls=':',
              label=r'$R \ll 1$')
ax[1].text(1*R_base + 0.00125, -25, r'$R \ll 1$', va='bottom', ha='left')


ax[1].axvline(n_gt*R_base, lw=2, color=fp.color(5, cmap='Grays', lut=7), ls=':',
              label=r'$R \gg 1$')
ax[1].text(n_gt*R_base - 30, -25, r'$R \gg 1$', va='bottom', ha='right')

ax[1].axvline(1, lw=1, color='0.5', ls='--')
ax[1].text(1.2, -25, r'$R = 1$', va='bottom', ha='left')
fp.add_labels(ax[1], r'Information ratio $R$ [unitless]',
              'BC-induced error 'r'[% ppb$^{-1}$]')

fp.add_legend(ax[1], bbox_to_anchor=(0.5, -0.35), loc='center', ncol=2)

ax[0].text(0.025, 0.85, '(a)', fontsize=ps.LABEL_FONTSIZE*ps.SCALE,
           transform=ax[0].transAxes, ha='left', va='bottom')
ax[1].text(0.025, 0.925, '(b)', fontsize=ps.LABEL_FONTSIZE*ps.SCALE,
           transform=ax[1].transAxes, ha='left', va='bottom')


ax[1].axhline(-6.912, lw=1, color='0.5', ls='--')
# fp.add_legend(ax[0], loc='lower right')


fig.subplots_adjust(hspace=0.4)

fp.save_fig(fig, plot_dir, 'limiting_analysis_2')

# %%
# import matplotlib as mpl
# cmap = mpl.colormaps['Grays'].resampled(7)
# cmap = mpl.colors.ListedColormap(cmap(np.linspace(0, 1, 7)[2:]))
# bounds = np.arange(1, 10 + 2, 2)*R_base
# norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
# cax = fp.add_cax(fig, ax[0], horizontal=True, cbar_pad_inches=0.75)
# cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
#                   cax=cax, orientation='horizontal',
#                   ticks=(bounds[:-1] + bounds[1:])/2)
# cb.ax[0].set_xticklabels([f'{m*R_base:.3f}' for m in range(1, 10, 2)])
# fp.format_cbar(cb, cbar_title=r'R [unitless]', horizontal=True)

# for i, m in enumerate(range(500, 1500, 200)):
#     ax[1].plot(xs + 0.5, R_gt_one(delta_c, k=tau_k, R=m*R_base)/delta_c, 
#                color=fp.color(i + 2, cmap='Grays', lut=7), lw=2)
#     ax[1].plot(xs + 0.5, preview_1d(delta_c, k=tau_k, sa=sa, so=so/m)/delta_c, 
#                    color=fp.color(i + 2, cmap='Grays', lut=7), lw=3, ls='--')

# bounds = np.round(np.arange(500, 1500 + 200, 200)*R_base, 2)
# norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
# cax = fp.add_cax(fig, ax[1], horizontal=True, cbar_pad_inches=0.75)
# cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
#                   cax=cax, orientation='horizontal', 
#                   ticks=(bounds[:-1] + bounds[1:])/2)
# cb.ax[0].set_xticklabels([f'{m*R_base:.2f}' for m in range(500, 1500, 200)])
# fp.format_cbar(cb, cbar_title=r'R [unitless]', horizontal=True)