"""
generate_figures.py
-------------------
Generates all mandatory figures for the report:
  1. Top-view contour plots with Nelder-Mead paths (n=2 only)
  2. Experimental convergence rate plots (all successful runs)

Authors: Elia Zonta, Giuseppe Fontanella, Fatima Ali
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
import os
from test_functions import ZakharovFunction, DixonPriceFunction, LevyFunction

FUNC_CLASSES = {
    'Zakharov':    ZakharovFunction,
    'Dixon-Price': DixonPriceFunction,
    'Levy':        LevyFunction,
}

_ROOT   = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(_ROOT, 'figures')
os.makedirs(OUT_DIR, exist_ok=True)

# ─── plotting style ──────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'text.usetex': False,
})

COLORS = [
    'tab:blue',    # x_bar
    'tab:orange',  # rand_1
    'tab:green',   # rand_2
    'tab:red',     # rand_3
    'tab:purple',  # rand_4
    'tab:brown',   # rand_5
]

# ─── helper: experimental convergence rate ────────────────────────────────────

def exp_conv_rate(f_history, f_star, tail=0.3):
    """
    Estimate experimental convergence order p from |f_k - f*|.
    Uses the last `tail` fraction of the error sequence.
    Returns array of local rates.
    """
    errors = np.abs(np.array(f_history) - f_star)
    # trim leading NaN/inf and trailing near-zero
    mask  = (errors > 1e-15) & np.isfinite(errors)
    if mask.sum() < 6:
        return None
    errors = errors[mask]
    # use only the tail for rate estimation
    start  = max(0, int(len(errors) * (1 - tail)))
    e      = errors[start:]
    if len(e) < 4:
        return None
    # If e_{k+1} ≈ C * e_k^p, then log(e_{k+1}) / log(e_k) ≈ p
    # when errors are small; this is used only as an empirical diagnostic
    rates = np.log(e[1:] + 1e-300) / (np.log(e[:-1] + 1e-300) + 1e-300)
    rates = rates[(rates > 0) & (rates < 5)]
    if len(rates) == 0:
        return None
    # Median is more robust than the mean against noisy local estimates.
    return float(np.median(rates))


# ═══════════════════════════════════════════════════════════════════════════════
#  FIGURE 1 — Top-view plots (n = 2)
# ═══════════════════════════════════════════════════════════════════════════════

def make_topview(fname, results_n2, func):
    """Contour map + NM paths for n=2."""
    # build grid
    cx, cy = func.x_bar[0], func.x_bar[1]
    span   = 2.5
    x1 = np.linspace(cx - span, cx + span, 400)
    x2 = np.linspace(cy - span, cy + span, 400)
    X1, X2 = np.meshgrid(x1, x2)
    Z = np.vectorize(lambda a, b: func(np.array([a, b])))(X1, X2)

    fig, ax = plt.subplots(figsize=(7, 6))

    # safe log scale for contours
    zmin = max(Z.min(), 1e-12)
    zmax = Z.max()
    if zmax > zmin:
        levels = np.logspace(np.log10(zmin), np.log10(zmax), 35)
    else:
        levels = 35
    cf = ax.contourf(X1, X2, Z, levels=levels, cmap='YlOrRd', alpha=0.55)
    ax.contour(X1, X2, Z, levels=levels, colors='gray', linewidths=0.4, alpha=0.6)
    plt.colorbar(cf, ax=ax, shrink=0.85, label='$f(x)$')

    labels = [r'$\bar{x}$'] + [f'rand {k+1}' for k in range(5)]

    for i, (res, col, lbl) in enumerate(zip(results_n2, COLORS, labels)):
        hist = res['history']
        # draw path
        ax.plot(hist[:, 0], hist[:, 1], '-', color=col,
                linewidth=1.2, alpha=0.85, zorder=3)
        # starting point
        ax.plot(hist[0, 0], hist[0, 1], 'o', color=col,
                markersize=7, zorder=5, label=lbl)
        # end point
        mk = '*' if res['success'] else 'x'
        ax.plot(hist[-1, 0], hist[-1, 1], mk, color=col,
                markersize=10, zorder=6, markeredgewidth=1.5)

    # global minimum
    ax.plot(func.x_star[0], func.x_star[1], 'k+',
            markersize=14, markeredgewidth=2.5, zorder=7, label='$x^*$')

    ax.set_xlabel('$x_1$'); ax.set_ylabel('$x_2$')
    ax.set_title(f'{fname} — Nelder-Mead paths ($n=2$)')
    ax.legend(loc='upper left', framealpha=0.9)
    ax.set_xlim(cx - span, cx + span)
    ax.set_ylim(cy - span, cy + span)

    fig.tight_layout()
    path = os.path.join(OUT_DIR, f'topview_{fname.lower().replace("-","_")}.pdf')
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved: {path}')


# ═══════════════════════════════════════════════════════════════════════════════
#  FIGURE 2 — Convergence rate plots (per function, per dimension)
# ═══════════════════════════════════════════════════════════════════════════════

def make_conv_plots(fname, results_all, func_class, dimensions):
    """One figure per dimension with all starting-point error curves."""

    for n in dimensions:
        func        = func_class(n)
        dim_results = results_all[n]

        fig, ax = plt.subplots(figsize=(7, 5))
        any_plotted = False

        labels = [r'$\bar{x}$'] + [f'rand {k+1}' for k in range(5)]

        for i, (res, col, lbl) in enumerate(zip(dim_results, COLORS, labels)):
            fh  = res['f_history']
            err = np.abs(np.array(fh) - func.f_star)
            err = np.maximum(err, 1e-18)

            mask  = err > 1e-15
            if mask.sum() < 3:
                continue
            iters = np.arange(len(err))
            ax.semilogy(iters, err, color=col, linewidth=1.2,
                        alpha=0.85, label=lbl)
            any_plotted = True

        if not any_plotted:
            plt.close(fig)
            continue

        ax.set_xlabel('Iteration $k$')
        ax.set_ylabel(r'$|f(x^{(k)}) - f^*|$')
        ax.set_title(f'{fname} — convergence ($n={n}$)')
        ax.legend(loc='upper right', framealpha=0.9)
        ax.grid(True, which='both', alpha=0.3)

        fig.tight_layout()
        path = os.path.join(OUT_DIR,
               f'conv_{fname.lower().replace("-","_")}_n{n}.pdf')
        fig.savefig(path, bbox_inches='tight')
        plt.close(fig)
        print(f'  saved: {path}')


# ═══════════════════════════════════════════════════════════════════════════════
#  FIGURE 3 — Local convergence rate estimates bar-chart (successful runs)
# ═══════════════════════════════════════════════════════════════════════════════

def make_rate_summary(fname, results_all, func_class, dimensions):
    """Bar chart of estimated convergence rates grouped by starting point."""
    fig, axes = plt.subplots(1, len(dimensions), figsize=(14, 4), sharey=True)
    labels = [r'$\bar{x}$'] + [f'r{k+1}' for k in range(5)]

    for ax, n in zip(axes, dimensions):
        func        = func_class(n)
        dim_results = results_all[n]
        rates = []
        cols  = []
        for res, col in zip(dim_results, COLORS):
            r = exp_conv_rate(res['f_history'], func.f_star)
            rates.append(r if r is not None else 0.0)
            cols.append(col)

        bars = ax.bar(labels, rates, color=cols, alpha=0.8, edgecolor='k', linewidth=0.5)
        ax.set_title(f'$n={n}$')
        ax.set_xlabel('Start point')
        ax.set_ylim(0, 2.0)
        ax.axhline(1.0, color='red', linestyle='--', linewidth=0.8, label='linear ($p=1$)')
        ax.grid(axis='y', alpha=0.3)
        if n == dimensions[0]:
            ax.set_ylabel('Est. rate $p$')
            ax.legend(fontsize=8)
        # annotate bars
        for bar, r in zip(bars, rates):
            if r > 0.05:
                ax.text(bar.get_x() + bar.get_width()/2, r + 0.03,
                        f'{r:.2f}', ha='center', va='bottom', fontsize=7)

    fig.suptitle(f'{fname} — Experimental convergence rates', fontsize=13)
    fig.tight_layout()
    path = os.path.join(OUT_DIR,
           f'rates_{fname.lower().replace("-","_")}.pdf')
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved: {path}')


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    with open(os.path.join(_ROOT, 'results.pkl'), 'rb') as fh:
        data = pickle.load(fh)

    results    = data['results']
    dimensions = data['dimensions']

    for fname, FuncClass in FUNC_CLASSES.items():
        print(f'\n{fname}')
        func2 = FuncClass(n=2)

        # 1 — top view
        make_topview(fname, results[fname][2], func2)

        # 2 — convergence curves
        make_conv_plots(fname, results[fname], FuncClass, dimensions)

        # 3 — rate summary
        make_rate_summary(fname, results[fname], FuncClass, dimensions)

    print('\nAll figures generated.')
