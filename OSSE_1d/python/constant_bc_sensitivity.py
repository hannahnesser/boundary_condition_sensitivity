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
np.set_printoptions(precision=3, linewidth=300, suppress=True)

## -------------------------------------------------------------------------##
# File Locations
## -------------------------------------------------------------------------##
project_dir, config = utils.setup()
data_dir = f'{project_dir}/data/data_OSSE'
plot_dir = f'{project_dir}/plots'
if not os.path.exists(plot_dir):
    os.mkdir(plot_dir)

## -------------------------------------------------------------------------##
# Constant BC perturbations
## -------------------------------------------------------------------------##
fig, ax = fp.get_figax(rows=1, cols=2, aspect=2, max_height=4.5, max_width=6.5,
                       width_ratios=[1, 1], sharey=True, sharex=False)
ax = ax.T.flatten()
ax[0].set_ylim(-0.22, 0.025)

# ax_t = [ax[0].twinx(), ax[1].twinx()]

# Define the base perturbation
pert = 10

# Create wind speed array
U = np.concatenate([np.arange(7, 3, -1), 
                    np.arange(3, 7, 1)])*24*60*60/1000
# U = np.repeat(U, 2)
Us = {'Constant' : 5*24*60*60/1000, 'Variable' : U}

for i, (wind_name, U) in enumerate(Us.items()):
    fp.add_title(ax[i], f'{wind_name}''\nwind speed')
    print('-'*70)
    print(wind_name)

    # BC perturbations
    # The first axis will plot the effect of BC perturbations on the
    # posterior solution 
    true_BC = inv.Inversion(U=U, gamma=1)
    print(true_BC.nobs)
    pert_BC = inv.Inversion(U=U, gamma=1, BC=true_BC.BCt + pert)
    print(f'  Average wind speed : {true_BC.U.mean()}')
    print(f'  DOFS: {np.trace(true_BC.a)}')

    # print(f'True Kx : {(true_BC.k @ true_BC.xa)[:5]}')
    # print(true_BC.k/true_BC.xa_abs)
    kx = true_BC.k @ true_BC.xa
    # print(f'True Kx mean : {kx.mean()}')

    # Plot the relative error in the posterior solution
    rrmse_true = ((pert_BC.xhat - true_BC.xhat)**2).mean()**0.5/pert_BC.xa.mean()/pert
    ax[i].plot(
        pert_BC.xp, 100*(pert_BC.xhat - true_BC.xhat)/true_BC.xa/pert,
        color='grey' , lw=5, label=f'True error')
    # ax[i].text(0.95, 0.65, f'RRMSE:', color='black',
    #            transform=ax[i].transAxes, ha='right', va='bottom')
    # ax[i].text(0.95, 0.55, f'{rrmse_true:.2f}', color='grey',
    #            transform=ax[i].transAxes, ha='right', va='bottom')

    # Plot approximations to estimate
    # ax[i].plot(pert_BC.xp + 0.04,
    #             100*pert_BC.preview_1d(pert)/pert,
    #             color=fp.color(1, cmap='plasma'), lw=4, ls=':', 
    #             zorder=20,
    #             label='1D preview metric')

    rrmse_preview = (pert_BC.preview_2d(pert)**2).mean()**0.5/pert_BC.xa.mean()/pert
    ax[i].plot(pert_BC.xp - 0.04,
                100*pert_BC.preview_2d(pert)/pert,
                color=fp.color(4, cmap='plasma'), lw=4, ls=':', 
                zorder=20,
                label='Preview')
    # ax[i].text(0.95, 0.45, f'{rrmse_preview:.2f}', 
    #            color=fp.color(4, cmap='plasma'),
    #            transform=ax[i].transAxes, ha='right', va='bottom')

    rrmse_diag = ((-pert*pert_BC.g.sum(axis=1))**2).mean()**0.5/pert_BC.xa.mean()/pert
    # ax[i].plot(pert_BC.xp,
    #             -100*pert*pert_BC.g.sum(axis=1)/pert_BC.xa/pert,
    #             color=fp.color(7, cmap='plasma'), lw=4, ls=':', 
    #             zorder=20,
    #             label='Diagnostic')
    # ax[i].text(0.95, 0.35, f'{rrmse_diag:.2f}', 
    #            color=fp.color(7, cmap='plasma'),
    #            transform=ax[i].transAxes, ha='right', va='bottom')
    
    # est_a = pert_BC.estimate_avker()
    # ax_t[i].plot(pert_BC.xp, est_a, color='red', ls='--', 
    #              label='Estimated averaging kernel')
    # ax_t[i].plot(pert_BC.xp, np.cumprod((1 - est_a)), color='blue', ls='-',
    #              label='Estimated product of (1 - a,j)')

    # BC correction
    # The third axis will plot the effect of correcting the boundary condition 
    # as part of the inversion
    # This isn't exactly insensitive to the magnitude of the perturbation, but oh
    # well
    # for i, pert in enumerate(perts):
    print('Boundary method : ')
    pert_opt_BC = inv.Inversion(
        U=U, gamma=1, BC=true_BC.BCt + pert, opt_BC=True)
    print(f'  Boundary condition correction: {pert_opt_BC.xhat_BC}')
    rrmse_boundary = ((pert_opt_BC.xhat - true_BC.xhat)**2).mean()**0.5/pert_opt_BC.xa.mean()/pert
    ax[i].plot(
        pert_opt_BC.xp, 100*(pert_opt_BC.xhat - true_BC.xhat)/true_BC.xa/pert, 
        color=fp.color(3, cmap='viridis'), lw=4, ls='--',
        label=f'Boundary method')
    # ax[i].text(0.95, 0.25, f'{rrmse_boundary:.2f}', 
    #            color=fp.color(3, cmap='viridis'),
    #            transform=ax[i].transAxes, ha='right', va='bottom')

    # print('Boundary method prior cost: ', pert_opt_BC.cost_prior())
    # ax_t[i].plot(
    #     pert_opt_BC.xp, -pert*pert_opt_BC.a_full[:-1, -1], color='blue', 
    #     label='A column'
    # )

    # Sa_abs scaling
    # The second axis will plot the effect of using a buffer grid cell
    # ax3 = ax[3].twinx()
    # ax3.set_ylim(-0.05, 0.05)
    # sfs = np.array([1, 5, 10, 50, 100])
    # for i, sf in enumerate(sfs):
    p = pert_BC.estimate_p(pert)
    print('Buffer method :')
    print(f'  Range of p values (buffer scaling): {p}')
    sa_sf = dc(true_BC.sa)
    sa_sf[0] *= p[-1]**2
    pert_sa_BC = inv.Inversion(
        U=U, sa=sa_sf, BC=true_BC.BCt + pert, gamma=1, opt_BC=False)
    rrmse_buffer = ((pert_sa_BC.xhat - true_BC.xhat)**2).mean()**0.5/pert_sa_BC.xa.mean()/pert
    # print('Buffer method prior cost: ', pert_sa_BC.cost_prior())
    ax[i].plot(
        pert_sa_BC.xp, 100*(pert_sa_BC.xhat - true_BC.xhat)/true_BC.xa/pert, 
        color=fp.color(7, cmap='plasma'), lw=4, ls='--',
        label='Buffer method')
    # ax[i].text(0.95, 0.15, f'{rrmse_buffer:.2f}', 
    #            color=fp.color(6, cmap='viridis'),
    #            transform=ax[i].transAxes, ha='right', va='bottom')

    # # # Correction and buffer (Sa_abs scaling)
    # # # The fourth axis will plot the effect of using a buffer grid cell
    # print('Combination method :')
    # sa_sf = dc(true_BC.sa)
    # sa_sf[0] *= sf**2
    # pert_sa_opt_BC = inv.Inversion(
    #     U=U, sa=sa_sf, BC=true_BC.BCt + pert, gamma=1, opt_BC=True)
    # print('Combined method prior cost: ', pert_sa_opt_BC.cost_prior())
    #     # Plot the relative error in the posterior solution
    #     ax[2].plot(
    #         pert_sa_opt_BC.xp, 
    #         stats.rel_err(pert_sa_opt_BC.xhat, true_BC.xhat),#/pert, 
    #         color=fp.color(8 - 2*i, lut=12), lw=4, label=f'{sf}')

    # fp.add_legend(ax[i], loc='upper left', 
    #             title='Corrected element', alignment='left',
    #             ncol=1, handlelength=2)
    # fp.add_legend(ax[5], loc='upper left', 
    #               title='Prior error scaling in buffer grid cell', 
    #               alignment='left', ncol=1, handlelength=1)

    # General formatting
    xmax = 13
    if i == 0:
        # ylabel = r'$\Delta \hat{x}/(x_A \sigma_c)$'
        ylabel = 'Boundary condition\ninduced error 'r'[% ppb$^{-1}$]'
        # xmax = 
    else:
        ylabel = ''
        # xmax = 7 #s.nstate

    ax[i].set_xticks(np.arange(0, xmax +  1, 1))
    ax[i].axhline(0, c='grey', alpha=0.2, zorder=-10)
    for k in range(xmax + 1):
        ax[i].axvline(k + 0.5, c=fp.color(1, lut=11), alpha=0.2,
                            ls=':', lw=0.5)
    xlabel = 'Grid cells from upwind boundary'
    fp.add_labels(ax[i], xlabel, ylabel)
    ax[i].set_xlim(0.5, xmax + 0.5)
# ax[0].set_xlim(0.5, 5.5)

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
plt.subplots_adjust(hspace=0.05)
ax[0].set_ylim(-0.25*100, 0.025*100)
# ax_t[0].set_ylim(-0.22, 0.025)
# ax_t[1].set_ylim(-0.22, 0.025)

ax[0].text(0.975, 0.05, '(a)', fontsize=ps.LABEL_FONTSIZE*ps.SCALE,
           transform=ax[0].transAxes, ha='right', va='bottom')
ax[1].text(0.975, 0.05, '(b)', fontsize=ps.LABEL_FONTSIZE*ps.SCALE,
           transform=ax[1].transAxes, ha='right', va='bottom')

fp.save_fig(fig, plot_dir, f'constant_BC')