"""
run_experiments.py
------------------
Runs the Nelder-Mead method on the three test functions for all required
dimensions and starting configurations, then saves all results to disk.

Random seed: SEED (minimum student ID of your team).

Authors: Elia Zonta, Giuseppe Fontanella, Fatima Ali
"""

import numpy as np
import pickle
import time
import os
from nelder_mead import nelder_mead
from test_functions import ALL_FUNCTIONS, ZakharovFunction, DixonPriceFunction, LevyFunction

# ─── configuration ────────────────────────────────────────────────────────────
SEED        = min(344867, 359949, 352152)          # students IDs min(344867-elia, 359949-Giuseppe, 352152-Fatima)
DIMENSIONS  = [2, 10, 20, 50]
N_RANDOM    = 5               # number of additional random starting points
MAX_ITER    = 50000           # max iterations per run

NM_PARAMS = dict(alpha=1.0, gamma=2.0, rho=0.5, sigma=0.5,
                 tol_f=1e-8, tol_x=1e-8)

# ─── helpers ──────────────────────────────────────────────────────────────────

def build_simplex(x0: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Build the initial simplex as required by the assignment:
      - vertex 0  : x0  (the given starting point)
      - vertex i+1: x0 + eta_i * e_i,  i = 1,...,n
    where eta_i is sampled uniformly from [0.5, 1] with a random sign ±1.
    """
    n       = len(x0)
    simplex = np.empty((n + 1, n), dtype=float)
    simplex[0] = x0.copy()
    for i in range(n):
        eta  = rng.uniform(0.5, 1.0)
        sign = rng.choice([-1.0, 1.0])
        simplex[i + 1]    = x0.copy()
        simplex[i + 1, i] = x0[i] + sign * eta
    return simplex


def run_all(seed=SEED, dimensions=DIMENSIONS, n_random=N_RANDOM,
            max_iter=MAX_ITER, nm_params=NM_PARAMS, verbose=True):
    """
    Run all experiments. Returns a nested dict:
        results[func_name][n] = list of per-run dicts
    Each per-run dict contains the output of nelder_mead() plus:
        'start_id'   : label ('x_bar' or 'rand_1' .. 'rand_5')
        'grad_norm'  : ||grad f(x_opt)|| computed with exact gradient
        'f_err'      : |f(x_opt) - f*|
    """
    results = {}
    rng_master = np.random.default_rng(seed)

    for FuncClass in ALL_FUNCTIONS:
        fname = FuncClass.name
        results[fname] = {}
        if verbose:
            print(f"\n{'='*60}")
            print(f"  Function: {fname}")
            print(f"{'='*60}")

        for n in dimensions:
            func  = FuncClass(n)
            x_bar = func.x_bar           # suggested starting point

            # ── generate 5 random starting points ──────────────────────────
            # each x_rand_i ~ Uniform on [x_bar - 1, x_bar + 1]^n
            random_starts = [
                x_bar + rng_master.uniform(-1.0, 1.0, size=n)
                for _ in range(n_random)
            ]
            all_starts  = [x_bar] + random_starts
            start_ids   = ['x_bar'] + [f'rand_{k+1}' for k in range(n_random)]

            dim_results = []
            if verbose:
                print(f"\n  n = {n}")

            for sp_id, x0 in zip(start_ids, all_starts):
                # build simplex with its own sub-RNG so simplex is reproducible
                func_idx = [FC.name for FC in ALL_FUNCTIONS].index(fname)
                sub_rng = np.random.default_rng(
                    seed + func_idx * 1000000 + n * 1000 + start_ids.index(sp_id)
                )
                simplex = build_simplex(x0, sub_rng)

                res = nelder_mead(func, simplex, max_iter=max_iter, **nm_params)

                # extra diagnostics
                res['start_id']  = sp_id
                res['x0']        = x0.copy()
                res['grad_norm'] = float(np.linalg.norm(func.gradient(res['x_opt'])))
                res['f_err']     = float(abs(res['f_opt'] - func.f_star))

                dim_results.append(res)

                if verbose:
                    icon = 'OK' if res['success'] else 'NO'
                    print(f"    [{icon}] {sp_id:8s}  iters={res['n_iter']:6d}"
                          f"  f_err={res['f_err']:.2e}"
                          f"  |grad|={res['grad_norm']:.2e}"
                          f"  t={res['time']:.2f}s")

            results[fname][n] = dim_results

    return results


# ─── entry point ──────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print(f"Random seed: {SEED}")
    print(f"Dimensions:  {DIMENSIONS}")
    print(f"NM params:   {NM_PARAMS}")

    t0      = time.perf_counter()
    results = run_all()
    total   = time.perf_counter() - t0

    out_path = os.path.join(os.path.dirname(__file__), 'results.pkl')
    with open(out_path, 'wb') as fh:
        pickle.dump({'results': results, 'seed': SEED,
                     'dimensions': DIMENSIONS, 'nm_params': NM_PARAMS}, fh)

    print(f"\nAll experiments done in {total:.1f}s  ->  saved to {out_path}")
