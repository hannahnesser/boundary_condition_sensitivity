import numpy as np
import pandas as pd
import math
from copy import deepcopy as dc
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Custom packages
import sys
sys.path.append('.')
import settings as s
import gcpy as gc
import inversion as inv
import format_plots as fp
import plot

plot_dir = '../plots'
plot_dir = f'{plot_dir}/n{s.nstate}_m{s.nobs}'

Cmax = 1
# U = np.concatenate([np.arange(5, 0, -1), 
#                     np.array([0.1, -0.1]), 
#                     np.arange(-1, -5, -1)])*24
# U = np.concatenate([U, U[::-1][1:-1]])
# U = np.repeat(U, 2)
U = np.array([5])*24
delta_t = np.abs(Cmax*s.L/U).min()
t = np.arange(0, s.init_t + s.total_t + delta_t, delta_t)
obs_t = t[t > s.init_t]
nobs_per_cell = len(obs_t)

# # Rescale U to be the length of the time array.
# repeat_factor = math.ceil(len(t)/len(U))
# U = np.tile(U, repeat_factor)[:len(t)]
# print(U[(t > s.init_t + delta_t)])

# Calculate the Courant number
C = U*delta_t/s.L

x = np.zeros(s.nstate)

bc_pert = 1900*np.ones(len(t))
bc_pert[(t > s.init_t + delta_t) & (t < s.init_t + 6 * delta_t)] = 2000

fm = inv.ForwardModel(
    nstate=s.nstate, nobs_per_cell=nobs_per_cell,
    Cmax=Cmax, L=s.L, U=U, 
    init_t=s.init_t, total_t=s.total_t, BC_t=s.BC_t, 
    x_abs_t=x)

y = fm.forward_model(x, bc_pert)
y = y.reshape((len(fm.y0), nobs_per_cell))

# Figure
fig, ax = fp.get_figax(max_width=10, max_height=9)
ax.set_ylim(1875, 2025)

line, = ax.plot(s.xp, y[:, 0], ls='--', color=fp.color(3))
label = ax.text(0.65, 2015, f't = {int(obs_t[0]*24)} hr', ha='left', va='top')
ax.text(20.5, 2015, r'C$_{max}$ ='f' {Cmax}', ha='right', va='top')
fig, ax = plot.format_plot(fig, ax, s.nstate)
ax = fp.add_labels(ax, 'State vector element', 'XCH4 (ppb)')

def animate(i):
    line.set_ydata(y[:, i])
    label.set_text(f't = {int(obs_t[i]*24)} hr')
    return line, label

ani = animation.FuncAnimation(fig, animate, frames=nobs_per_cell, blit=True)
# Writer = animation.writers['ffmpeg']
writer = animation.FFMpegWriter(fps=10, bitrate=800)

# writer = animation.PillowWriter(fps=10,
#                                 # metadata=dict(artist='Me'),
#                                 bitrate=1800)
plt.tight_layout()
ani.save(f'{plot_dir}/advect_pulse_c{Cmax}.mp4', writer=writer,
         dpi=500)
plt.close()