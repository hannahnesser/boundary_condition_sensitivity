# ## -------------------------------------------------------------------------##
# # Solve the inversion with the "true" boundary condition
# ## -------------------------------------------------------------------------##
# y_a = fm.forward_model(x_a, y_init, BC_t, t, U, L, obs_t)
# inv_inputs = [x_a, s_a_vec, y.flatten(), y_a.flatten(), s_o_vec, K_t*x_a]
# x_hat_t, s_hat, a_t, g_t = inv.solve_inversion(*inv_inputs, optimize_BC)
# y_hat_t = fm.forward_model(x_hat_t*x_a, y_init, BC_t, t, U, L, obs_t)
# bw_base = inv.band_width(g_t*x_a.reshape(-1, 1)*3600*24)
# ils_base = inv.influence_length(g_t*x_a.reshape(-1, 1)*3600*24)
# base = [bw_base, bw_base, ils_base, ils_base]


import matplotlib.pyplot as plt
import numpy as np

x = np.arange(0, 10, 0.01)
plt.plot(x, np.sin(4*x))
plt.plot(x, np.sin(4*x + np.pi/2))
plt.xlim(0, 10)
plt.ylim(-1.01, 1.01)
plt.show()

## -------------------------------------------------------------------------##
# Oscillating boundary condition perturbations
## -------------------------------------------------------------------------##
optimize_BC = False
# Summary plot
fig_perts, ax_perts = fp.get_figax(aspect=5)
fig_summ, ax_summ = plot.format_plot(nstate)
fig_scat, ax_scat = fp.get_figax(aspect=1)
# # Add ~10% errors
ax_summ.fill_between([0, nstate+1], 10, color='grey', alpha=0.2,
                     label=r'$\approx$ 10\% error')

# Iteration through perturbations
# BC = [vertical_shift, amplitude, frequency, phase shift]
# (pi halves should be a cosine)
BC1 = [1975, -100, 2, 0]
BC2 = [1975, 50, 2, 0]
BC3 = [1975, 100, 2, 0]
BC4 = [1975, -100, 2, np.pi/2]
BC5 = [1975, -100, 1, 0]
BC6 = [1975, -100, 4, 0]
# BC5 = [1975, -100, 6]
# # BC7 = [1975, -50, 2]
BCs = [BC1, BC2, BC3, BC4, BC5, BC6]
# BCs = [BC5]
# BCs = [BC1]
ls = ['-', '-']
x_hat_diff_record = np.zeros((2, len(BCs), nstate))
for i, BC_l in enumerate(BCs):
    for j, optimize_BC in enumerate([False]):#enumerate([True, False]):
    # for i, BC_l in enumerate([BC1]):
        n = len(BCs)
        BC = BC_l[0] + BC_l[1]*np.sin(BC_l[2]*2*np.pi/t.max()*t+BC_l[3])

        # Label
        pi_unicode = "\u03C0"
        alpha_unicode = "\u03B1"
        if BC_l[3] > 0:
            label = r'%d + %d sin(%d$\alpha$t + $\pi$/2)' % (int(BC_l[0]), int(BC_l[1]), int(BC_l[2]))
        else:
            label = r'%d + %d sin(%d$\alpha$t)' % (int(BC_l[0]), int(BC_l[1]), int(BC_l[2]))

        # Solve inversion
        K = inv.build_jacobian(x_a, y_init, BC, t, U, L, obs_t, optimize_BC)
        y_a = fm.forward_model(x_a, y_init, BC, t, U, L, obs_t)
        inv_inputs = [x_a, s_a_vec, y.flatten(), y_a.flatten(), s_o_vec, K]
        x_hat, s_hat, a, g = inv.solve_inversion(*inv_inputs, optimize_BC)
        x_hat_diff = np.abs(x_hat - x_hat_t)*x_a*3600*24
        x_hat_diff_record[j, i, :] = x_hat_diff
        y_hat = fm.forward_model(x_hat*x_a, y_init, BC, t, U, L, obs_t)

        c = y_a.flatten() - K @ np.ones(K.shape[1])
        delta_c = c - c_t
        x_hat_diff_pred = np.array(-(g*x_a.reshape((-1, 1))*3600*24) @ delta_c)


        # print('-'*70)
        # print(np.abs(g.sum(axis=1))*x_a*3600*24)
        # print(c)
        c = c.reshape(y_a.shape)
        # print(c)

        # Plots
        title = f'Oscillating Boundary Condition\n(BC = {int(BC_l[0]):d} + {int(BC_l[1]):d}sin({int(BC_l[2]):d}at) ppb)'
        # Perturbation plot
        if j == 0:
            ax_perts.plot(t/3600, BC, c=fp.color(i*2, lut=n*2), lw=2,
                          label=label)

        # # Inversion plot
        # fig, ax = plot.plot_inversion(x_a, x_hat, x_t, x_hat_t,
        #                               optimize_BC=optimize_BC)
        # ax = fp.add_title(ax, title)
        # fp.save_fig(fig, plot_dir, f'oscillating_BC_{(i+1)}_{optimize_BC}')

        # # Plot observations
        # fig, ax = plot.plot_obs(nstate, y, y_a, y_init, obs_t, optimize_BC)
        # ax = fp.add_title(ax, title)
        # fp.save_fig(fig, plot_dir, f'oscillating_BC_{(i+1)}_{optimize_BC}_obs')

        # fig, ax = plot.plot_obs_diff(nstate, y, y_hat, y_a, obs_t, optimize_BC)
        # ax = fp.add_title(ax, title)
        # fp.save_fig(fig, plot_dir, f'oscillating_BC_{(i+1)}_{optimize_BC}_obs_diff')

        # Summary plot
        ax_summ.plot(xp, x_hat_diff, c=fp.color(i*2, lut=n*2), lw=1, ls=ls[j],
                     label=label)
        # ax_summ.plot(xp, x_hat_diff_pred, c=fp.color(i*2, lut=n*2),
        #              lw=3, ls='--')
        # ax_summ.plot(xp, x_hat*x_a*3600*24, c=fp.color(i*2, lut=n*2),
        #              lw=1, ls=':')

        print('-'*30)
        print(np.where(np.abs(g.sum(axis=1)*x_a*3600*24*100) > 10))
        xx = (x_hat*x_a*3600*24)[:4].reshape((-1, 1))
        gg = (g.sum(axis=1)*x_a*3600*24)[:4].reshape((-1, 1))
        # m, b, r, bias = gc.comparison_stats(gg, xx)
        m, _, _, _ = np.linalg.lstsq(gg, xx, rcond=None)
        BC_err1 = -m[0][0]

        m, _, _, _ = np.linalg.lstsq(xx, gg, rcond=None)
        BC_err2 = -1/m[0][0]

        m = gc.rma(gg, xx)
        BC_err3 = -m

        m = gc.rma_modified(gg, xx)
        BC_err4 = -m

        print('-'*50)
        print(f'Predicted boundary condition error (RMA)         : {BC_err3:.1f}')
        print(f'Predicted boundary condition error (RMA modified): {BC_err4:.1f}')
        print(f'Predicted boundary condition error (LSQ, x vs. g): {BC_err1:.1f}')
        print(f'Predicted boundary condition error (LSQ, g vs. x): {BC_err2:.1f}')
        print('-'*50)

        # ax_scat.scatter(xx, gg, color=fp.color(i*2, lut=n*2), label=label)
        # ax_scat.scatter(xx, gg/m, color=fp.color(i*2, lut=n*2),
        #                 s=25, marker='x')
        ax_scat.scatter(xx, -BC_err1*gg, color=fp.color(i*2, lut=n*2),
                        s=10, marker='o')
        ax_scat.scatter(xx, -BC_err2*gg, color=fp.color(i*2, lut=n*2),
                        s=10, marker='v')
        ax_scat.scatter(xx, -BC_err3*gg, color=fp.color(i*2, lut=n*2),
                        s=20, marker='x')
        ax_scat.scatter(xx, -BC_err4*gg, color=fp.color(i*2, lut=n*2),
                        s=20, marker='*')

        ax_summ.plot(xp, np.abs(BC_err3*g.sum(axis=1)*x_a*3600*24),
                     c=fp.color(i*2, lut=n*2), lw=1, ls='--')
        # ax_summ.plot(xp,
        #              np.abs(x_hat - (1/m)*g.sum(axis=1) - x_hat_t)*x_a*3600*24,
        #              c=fp.color(i*2, lut=n*2), lw=1, ls=':')
        ax_summ.plot(xp,
                     np.abs(x_hat + BC_err3*g.sum(axis=1) - x_hat_t)*x_a*3600*24,
                     c=fp.color(i*2, lut=n*2), lw=1, ls='-.')

        # ax_summ.plot(xp, np.abs((1/m2)*g.sum(axis=1)*x_a*3600*24),
        #              c=fp.color(i*2, lut=n*2), lw=1, ls=':')



# styles = ['--', '-.', ':']
# # colors = ['0', '0.2', '0.4']
# perts = [25, 50, 100]
# for i, pert in enumerate(perts):
#     ax_summ.plot(xp, np.abs(-pert*g.sum(axis=1))*x_a*3600*24,
#                  c='black', lw=1, ls=styles[i],
#                  label=f'+/-{pert:d} ppb')
# ax_summ.fill_between(xp, np.sqrt(s_a_vec)*x_a*3600*24, color='grey', alpha=0.1,
#                      label='Prior error')
# ax_summ.fill_between(x_p, np.sqrt(np.diag(s_hat))*x_a*3600*24, color='grey', alpha=0.1)


# Perturbation summary aesthetics
ax_perts.axhline(s.BC_abs_t, color='grey', ls='--', lw=2,
                 label='True boundary condition')
ax_perts.axvspan(obs_t.min()/3600, obs_t.max()/3600, color='grey', alpha=0.1,
                 label='Observation times')
ax_perts.set_xlim(t.min()/3600, t.max()/3600)
ax_perts.set_ylim(1700, 2300)
ax_perts = fp.add_title(ax_perts,
                        'Oscillating Boundary Condition Perturbations')
ax_perts = fp.add_labels(ax_perts, 'Time (hr)', 'BC (ppb)')
fp.add_legend(ax_perts, bbox_to_anchor=(0.5, -0.45), loc='upper center',
              ncol=2)
fp.save_fig(fig_perts, plot_dir, f'oscillating_BC_perts_summary_{optimize_BC}')

## Summary aesthetics
fp.add_title(ax_summ, 'Oscillating Boundary Condition Perturbations')
# plot.add_text_label(ax_summ, optimize_BC)
fp.add_labels(ax_summ, 'State vector element',
              r'$\vert\Delta\hat{x}\vert$ (ppb/day)')

# Set limits
ax_summ.set_ylim(0, 100)

# Add legend
custom_lines = [Line2D([0], [0], color='grey', lw=1, ls='-'),
                Line2D([0], [0], color='grey', lw=2, ls='--')]
custom_labels = ['BC not optimized', 'BC optimized']
handles, labels = ax_summ.get_legend_handles_labels()
custom_lines.extend(handles)
custom_labels.extend(labels)
fp.add_legend(ax_summ, handles=custom_lines, labels=custom_labels,
              bbox_to_anchor=(0.5, -0.45), loc='upper center', ncol=2)

# Save
fp.save_fig(fig_summ, plot_dir, f'oscillating_BC_summary_{optimize_BC}')

# Scatter plot
fp.add_title(ax_scat, r'$\sum$G vs. $\hat{x}$')
fp.add_labels(ax_scat, r'$\hat{x}$ (ppb/day)', r'$\sum$G (ppb/day)')
fp.plot_one_to_one(ax_scat)
fp.save_fig(fig_scat, plot_dir, f'scatter')

# One last plot
fig, ax = plot.format_plot(nstate)
for i, BC_l in enumerate(BCs):
    if BC_l[3] > 0:
        label = r'%d + %d sin(%d$\alpha$t + $\pi$/2)' % (int(BC_l[0]), int(BC_l[1]), int(BC_l[2]))
    else:
        label = r'%d + %d sin(%d$\alpha$t)' % (int(BC_l[0]), int(BC_l[1]), int(BC_l[2]))

    d = x_hat_diff_record[0, i, :] - x_hat_diff_record[1, i, :]
    ax.plot(xp, d, c=fp.color(i*2, lut=n*2), lw=2, ls=ls[j], label=label)
ax.set_ylim(-10, 10)
ax.axhline(0, ls='--', color='grey')
fp.save_fig(fig, plot_dir, 'optimize_BC_difference')


plt.close()
