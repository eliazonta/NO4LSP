"""
nelder_mead.py
--------------
Implementation of the Nelder-Mead simplex method for unconstrained optimization.

The algorithm follows the classical Nelder & Mead (1965) formulation with the
four operations: reflection, expansion, contraction, and shrink.

Parameters (recommended defaults):
    alpha  = 1.0  (reflection)
    gamma  = 2.0  (expansion)
    rho    = 0.5  (contraction)
    sigma  = 0.5  (shrink)

Stopping criteria (both must be satisfied):
    |f_worst - f_best| < tol_f   (function value spread)
    max_i ||x_i - x_best|| < tol_x  (simplex diameter)

Author: Elia Zonta
"""

import numpy as np
import time


def nelder_mead(f, simplex0, alpha=1.0, gamma=2.0, rho=0.5, sigma=0.5,
                tol_f=1e-8, tol_x=1e-8, max_iter=50000):
    """
    Nelder-Mead simplex method.

    Parameters
    ----------
    f        : callable, objective function f: R^n -> R
    simplex0 : ndarray of shape (n+1, n), initial simplex vertices (rows)
    alpha    : float, reflection coefficient (> 0)
    gamma    : float, expansion coefficient (> 1)
    rho      : float, contraction coefficient (0 < rho < 1)
    sigma    : float, shrink coefficient (0 < sigma < 1)
    tol_f    : float, tolerance on function value range
    tol_x    : float, tolerance on simplex diameter
    max_iter : int,   maximum number of iterations

    Returns
    -------
    dict with keys:
        x_opt    : best point found
        f_opt    : function value at x_opt
        n_iter   : number of iterations performed
        success  : bool, True if converged within max_iter
        time     : elapsed wall-clock time (seconds)
        history  : list of best vertex at each iteration (for convergence plots)
        f_history: list of f(best vertex) at each iteration
    """
    # ------------------------------------------------------------------ init
    simplex = np.array(simplex0, dtype=float)   # shape (n+1, n)
    n = simplex.shape[1]
    assert simplex.shape[0] == n + 1, "Initial simplex must have n+1 rows"

    # Evaluate function at all vertices
    f_vals = np.array([f(simplex[i]) for i in range(n + 1)], dtype=float)

    history   = []   # best vertex at each iteration
    f_history = []   # best f value at each iteration

    t_start = time.perf_counter()

    for k in range(max_iter):
        # ---------------------------------------------------------- sort
        order  = np.argsort(f_vals)
        simplex = simplex[order]
        f_vals  = f_vals[order]

        x_best  = simplex[0];  f_best  = f_vals[0]
        x_worst = simplex[-1]; f_worst = f_vals[-1]
        x_2worst            = simplex[-2]; f_2worst = f_vals[-2]

        history.append(x_best.copy())
        f_history.append(f_best)

        # -------------------------------------------- convergence check
        f_range      = abs(f_worst - f_best)
        simplex_diam = np.max(np.linalg.norm(simplex[1:] - x_best, axis=1))

        if f_range < tol_f and simplex_diam < tol_x:
            elapsed = time.perf_counter() - t_start
            return {
                'x_opt':    x_best,
                'f_opt':    f_best,
                'n_iter':   k + 1,
                'success':  True,
                'time':     elapsed,
                'history':  np.array(history),
                'f_history': np.array(f_history),
            }

        # -------------------------------------------- centroid (all but worst)
        centroid = np.mean(simplex[:-1], axis=0)

        # -------------------------------------------- reflection
        x_r = centroid + alpha * (centroid - x_worst)
        f_r = f(x_r)

        if f_best <= f_r < f_2worst:
            # Accept reflection
            simplex[-1] = x_r
            f_vals[-1]  = f_r

        elif f_r < f_best:
            # ---------------------------------------- expansion
            x_e = centroid + gamma * (x_r - centroid)
            f_e = f(x_e)
            if f_e < f_r:
                simplex[-1] = x_e
                f_vals[-1]  = f_e
            else:
                simplex[-1] = x_r
                f_vals[-1]  = f_r

        else:
            # ---------------------------------------- contraction
            if f_r < f_worst:
                # Outside contraction
                x_c = centroid + rho * (x_r - centroid)
                f_c = f(x_c)
                if f_c <= f_r:
                    simplex[-1] = x_c
                    f_vals[-1]  = f_c
                else:
                    # -------------------------------- shrink
                    for i in range(1, n + 1):
                        simplex[i] = x_best + sigma * (simplex[i] - x_best)
                        f_vals[i]  = f(simplex[i])
            else:
                # Inside contraction
                x_c = centroid + rho * (x_worst - centroid)
                f_c = f(x_c)
                if f_c < f_worst:
                    simplex[-1] = x_c
                    f_vals[-1]  = f_c
                else:
                    # -------------------------------- shrink
                    for i in range(1, n + 1):
                        simplex[i] = x_best + sigma * (simplex[i] - x_best)
                        f_vals[i]  = f(simplex[i])

    # ------------------------------------------------- did not converge
    elapsed = time.perf_counter() - t_start
    order   = np.argsort(f_vals)
    return {
        'x_opt':     simplex[order[0]],
        'f_opt':     f_vals[order[0]],
        'n_iter':    max_iter,
        'success':   False,
        'time':      elapsed,
        'history':   np.array(history),
        'f_history': np.array(f_history),
    }
