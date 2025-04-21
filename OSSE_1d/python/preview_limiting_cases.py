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
data_dir = f'{project_dir}/data/data_OSSE'
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


from matplotlib import cm

def preview_1d(sa_bc, k, sa, so, n=20):    
    # Define secondary variables
    R = (so/(k*sa))**2
    k = k*np.tril(np.ones((n, n)))
    so_inv = np.diag(1/(so**2*np.ones(n)))
    sa_inv = np.diag(1/(sa**2*np.ones(n)))
    g = np.linalg.inv(sa_inv + k.T @ so_inv @ k) @ k.T @ so_inv
    return -sa_bc*g.sum(axis=1)

def lower_bound(sa_bc, k, R=1e-10, n=20):
    j = np.arange(1, n + 1)
    return -(sa_bc/k)*R**(j - 1)

def upper_bound(sa_bc, k, R=1e10, n=20):
    j = np.arange(1, n + 1)
    print(n - j + 1)
    return -(sa_bc/k)*R**(-1)*(n - j + 1)

ks = np.arange(0.05, 2, 0.1)
Rs = []
prevs = []
kk = 0.4
for sasa in np.arange(0.1, 2.25, 0.25): # Assuming xa = 30
    for soso in np.arange(5, 21, 2):
        Rs.append((soso/(kk*sasa*30))**2)
        prevs.append(preview_1d(10, kk, sasa*30, soso))

Rs = np.array(Rs)
prevs = np.array(prevs)

fig, ax = fp.get_figax()
for i in range(prevs.shape[0]):
    ax.plot(np.arange(1, 21), prevs[i, :], c=cm.hot(Rs[i]/Rs.max()))
ax.plot(np.arange(1, 21), -lower_bound(10, k=kk, R=Rs.min()), color='red')
ax.plot(np.arange(1, 21), -upper_bound(10, k=kk, R=Rs.max()), color='blue')

# ax.plot(upper_bound(10, k=ks[0], R=Rs[0]), color='blue')
# %%
