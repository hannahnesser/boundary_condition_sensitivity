import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
from collections import OrderedDict
import inversion as inv

cmap = colormaps['plasma']

## -------------------------------------------------------------------------##
# Demonstrate the use of the Inversion class
## -------------------------------------------------------------------------##
# An inversion with all of the defaults.
orig = inv.Inversion()

# Now try varying the prior error (let's double it).
var_sa = inv.Inversion(sa=orig.sa*4)

# Or, we can change the wind speed. This will generate almost an entirely
# different OSSE because the pseudo-observations will be different.
var_u = inv.Inversion(U=7*24)

# Maybe we want transport to be worse, so we can lower the Courant number.
var_c = inv.Inversion(Cmax=0.5)

# Let's perturb the boundary condition by some oscillating function.
bc_pert = orig.BC + 10*np.sin((2*np.pi/orig.t.max())*2*orig.t)
var_bc = inv.Inversion(BC=bc_pert)

## -------------------------------------------------------------------------##
# Plot the results
## -------------------------------------------------------------------------##
fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

# First, plot the original inversion along with the prior and true emissions
ax[0].plot(orig.xp, orig.xt_abs, c='black')
ax[0].plot(orig.xp, orig.xa_abs,
            c=cmap(1), marker='.', markersize=10,
            label=r'Prior($\pm$ 50\%)')
ax[0].fill_between(orig.xp, 
                   orig.xa_abs - orig.xa_abs*orig.sa**0.5,
                   orig.xa_abs + orig.xa_abs*orig.sa**0.5,
                   color=cmap(1), alpha=0.2, zorder=-1)

# Next, plot our examples.
examples = {'Double the prior errors' : var_sa,
            'Increase wind speed' : var_u,
            'Lower the Courant number' : var_c,
            'Oscillating boundary condition' : var_bc}
for i, (label, ex) in enumerate(examples.items()):
    ax[0].plot(ex.xp, ex.xhat*ex.xa_abs, c=cmap((2 + i)/5), ls='--', label=label)
handles, labels = ax[0].get_legend_handles_labels()

# Finally, plot the observations
ax[1].plot(orig.xp, orig.y0, c='black', label='Steady state', zorder=10)
ax[1].plot(orig.xp, orig.y.reshape(orig.nobs_per_cell, orig.nstate).T,
           c='grey', label='Observations\n($\pm$ 15 ppb)',
           lw=0.5, zorder=9)
y_err_min = (orig.y.reshape(orig.nobs_per_cell, orig.nstate).T - 
             orig.so.reshape(orig.nobs_per_cell, orig.nstate).T**0.5).min(axis=1)
y_err_max = (orig.y.reshape(orig.nobs_per_cell, orig.nstate).T + 
             orig.so.reshape(orig.nobs_per_cell, orig.nstate).T**0.5).max(axis=1)
ax[1].fill_between(orig.xp, y_err_min, y_err_max, color='grey', alpha=0.2)
handles1, labels1 = ax[1].get_legend_handles_labels()

# Aesthetics
## Legend
handles.extend(handles1)
labels.extend(labels1)
labels = OrderedDict(zip(labels, handles))
handles = labels.values()
labels = labels.keys()
ax[1].legend(handles=handles, labels=labels,
             bbox_to_anchor=(0.9, 0.5), loc='center left', ncol=1,
             bbox_transform=fig.transFigure)

## Limits and labels
ax[0].set_xlabel('')
ax[0].set_ylabel('Emissions\n(ppb/day)')
ax[0].set_xlim(orig.xp.min() - 0.5, orig.xp.max() + 0.5)
ax[0].set_ylim(0, 50)

ax[1].set_xlabel('State vector element')
ax[1].set_ylabel('XCH4\n(ppb)')
ax[1].set_ylim(1890, 2050)
ax[1].set_xticks(orig.xp, orig.xp)

# Save out
fig.savefig(f'demo.png', bbox_inches='tight', transparent=True)