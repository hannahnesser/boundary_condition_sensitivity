import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.lines import Line2D
import numpy as np

import sys
sys.path.append('.')
import format_plots as fp
import config

## -------------------------------------------------------------------------##
# Define plotting functions
## -------------------------------------------------------------------------##
def plot_inversion(x_a, x_hat, x_true, x_hat_true=None, s_a=None, a=None,
                   optimize_BC=False):
    # Set up plots
    nstate = x_a.shape[0]
    fig, ax = format_plot(nstate)

    # Subset prior error
    if optimize_BC and (s_a is not None):
        s_a = s_a[:-1, :-1]

    # Add text
    add_text(ax, optimize_BC)

    # Get plotting x coordinates
    xp = np.arange(1, nstate+1)

    # Plot "true " emissions
    ax.plot(xp, 3600*24*x_true, c=fp.color(2), ls='--', label='Truth')

    # Plot prior
    if s_a is None:
        ax.plot(xp, 3600*24*x_a, c=fp.color(4), marker='.', markersize=10,
                label='Prior')
    else:
        ax.errorbar(xp, 3600*24*x_a, yerr=3600*24*np.diag(s_a)**0.5*x_a,
                    c=fp.color(4), marker='.', markersize=10,
                    label='Prior')

    # Pot posterior
    if x_hat_true is not None:
        ax.plot(xp, 3600*24*x_hat_true*x_a, c=fp.color(6),
                marker='*', markersize=10, label='True BC Posterior')
        ax.plot(xp, 3600*24*x_a*x_hat, c=fp.color(8), marker='.', markersize=5,
                lw=1, label='Posterior')
        ncol = 2
    else: # if x_hat_true is none
        ax.plot(xp, 3600*24*x_a*x_hat, c=fp.color(6),
                marker='*', markersize=10,
                label='True BC Posterior')

        ncol = 3

    ax = fp.add_legend(ax, bbox_to_anchor=(0.5, -0.35),
                       loc='upper center', ncol=ncol)
    ax = fp.add_labels(ax, 'State Vector Element', 'Emissions\n(ppb/day)')
    ax.set_ylim(0, 200)

    return fig, ax

def plot_avker(nstate, a):
    fig, ax = format_plot(nstate, nplots=1)
    # add_text(ax)

    # Get plotting x coordinates
    xp = np.arange(1, nstate+1)

    # Plot avker sensitivities
    ax.plot(xp, np.diag(a), c='grey', lw=1)
    for i, row in enumerate(a):
        ax.plot(xp, row, c=fp.color(i*2, lut=a.shape[0]*2))

    # Add text
    fp.add_labels(ax, 'State Vector Element', 'Averaging kernel\nsensitivities')

    # Set limits
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.5, 1])
    return fig, ax

def plot_obs(nstate, y, y_a, y_ss, obs_t, optimize_BC):
    # Plot observations
    fig, ax = format_plot(nstate)

    xp = np.arange(1, nstate+1)
    nt = len(obs_t)
    ax.plot(xp, y_ss, c='black', lw=2, label='Steady State', zorder=10)
    ax.plot(xp, y, c='grey', lw=0.5, ls=':', markersize=10,
            label='Observed', zorder=9)
    for i, y_a_column in enumerate(y_a.T):
        if (i == 0) or (i == (nt - 1)):
            ax.plot(xp, y_a_column,
                    c=fp.color(i+2, cmap='plasma_r', lut=nt + 2),
                    lw=0.5, ls='-',
                    label=f'Modeled (t={(obs_t[i]/3600):.1f} hrs)')
        else:
            ax.plot(xp, y_a_column,
                    c=fp.color(i+2, cmap='plasma_r', lut=nt + 2),
                    lw=0.5, ls='-')

    # Aesthetics
    add_text(ax, optimize_BC)
    ax = fp.add_legend(ax, bbox_to_anchor=(0.5, -0.35), loc='upper center',
                       ncol=2)
    ax = fp.add_labels(ax, 'State Vector Element', 'XCH4 (ppb)')
    ax.set_ylim(y_ss.min()-50, y_ss.max()+50)

    return fig, ax

def plot_obs_diff(nstate, y, y_hat, y_a, obs_t, optimize_BC):
    # Plot observations
    fig, ax = format_plot(nstate)

    xp = np.arange(1, nstate+1)
    nt = len(obs_t)
    ax.axhline(0, ls='--', lw=1, color='grey')
    y_diff = y - y_a
    for i, y_column in enumerate(y_diff.T):
        if (i == 0) or (i == (nt - 1)):
            ax.plot(xp, y_column,
                    c=fp.color(i+2, cmap='plasma', lut=nt + 2),
                    lw=0.5, ls='-',
                    label=f'Difference (t={(obs_t[i]/3600):.1f} hrs)')
        else:
            ax.plot(xp, y_column,
                    c=fp.color(i+2, cmap='plasma', lut=nt + 2),
                    lw=0.5, ls='-')

    y_diff = y - y_hat
    for i, y_column in enumerate(y_diff.T):
        ax.plot(xp, y_column,
                c=fp.color(i+2, cmap='viridis', lut=nt + 2),
                lw=0.5, ls='--')

    # Aesthetics
    add_text(ax, optimize_BC)
    ax = fp.add_legend(ax, bbox_to_anchor=(0.5, -0.35), loc='upper center',
                       ncol=2)
    ax = fp.add_labels(ax, 'State Vector Element', 'XCH4 (ppb)')
    ax.set_ylim(-200, 200)

    return fig, ax

def plot_cost_func(x_hat, x_a, s_a_vec, y_hat, y, s_o_vec, obs_t,
                   optimize_BC):
    # Plot observations
    nstate = len(x_a)
    nobs = len(s_o_vec)
    fig, ax = format_plot(nstate)
    # ax2 = ax.twinx()

    xp = np.arange(1, nstate+1)
    nt = len(obs_t)

    # Plot x component
    x_diff = (x_hat - np.ones(nstate))**2/s_a_vec
    ax.plot(xp, x_diff, c=fp.color(5), lw=2, ls='-', label=r'$J_A(\hat{x})$')
    #, color=fp.color(5))
    # ax.tick_params(axis='y', labelcolor=fp.color(5))
    # ax.set_ylim(0, 50) # x_diff.max()*1.1)

    # Plot y component
    y_diff = ((y - y_hat)**2/s_o_vec.reshape(y.shape))#.sum(axis=1)
    ax.plot(xp, y_diff.sum(axis=1), c=fp.color(2), lw=2, ls='-',
            label=r'$J_O(\hat{x})$')
    for i, y_column in enumerate(y_diff.T):
        ax.plot(xp, y_column, c=fp.color(2), lw=0.5, ls='--')


    # y_diff = (y - y_a)**2/s_o_vec.reshape(y.shape)/nobs
    # ax2.plot(xp, y_diff.sum(axis=1), c=fp.color(2), lw=2, ls='--')

    # ax2 = fp.add_labels(ax2, '', r'$J_O(\hat{x})$', color=fp.color(2))
    # ax2.tick_params(axis='y', labelcolor=fp.color(2))
    # ax2.set_ylim(0, y_diff.sum(axis=1).max()*1.1)

    # Aesthetics
    add_text(ax, optimize_BC)
    ax = fp.add_labels(ax, '', r'$J(\hat{x})$')
    ax.set_xlabel('State Vector Element',
                  fontsize=config.LABEL_FONTSIZE*config.SCALE,
                  labelpad=config.LABEL_PAD, color='black')
    ax = fp.add_legend(ax, bbox_to_anchor=(0.5, -0.35), loc='upper center',
                       ncol=2)
    ax.set_ylim(0, 30)
    # ax.set_ylim(0, 5)

    return fig, ax

def format_plot(nstate, nplots=1, **fig_kwargs):
    fig, ax = fp.get_figax(aspect=4 - 0.65*(nplots-1), cols=1, rows=nplots,
                           **fig_kwargs)
    if nplots == 1:
        ax = [ax]
    for axis in ax:
        for i in range(nstate+2):
            axis.axvline(i-0.5, c=fp.color(1), alpha=0.2, ls=':')
        axis.set_xticks(np.arange(0, nstate+1, 2))
        axis.set_xlim(0.5, nstate+0.5)
        axis.set_facecolor('white')
    if nplots == 1:
        return fig, ax[0]
    else:
        return fig, ax

def add_text(ax, optimize_BC):
    if optimize_BC:
        txt = 'BC optimized'
    else:
        txt = 'BC not optimized'
    # txt = txt + f'\nn = {nstate}\nm = {nobs}\nU = {(U*3600)}'
    ax.text(0.98, 0.95, txt, ha='right', va='top',
                 fontsize=config.LABEL_FONTSIZE*config.SCALE,
                 transform=ax.transAxes)
