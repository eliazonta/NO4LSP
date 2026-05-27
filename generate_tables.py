r"""
generate_tables.py
------------------
Reads results.pkl and prints LaTeX table rows (Table 1 format from the
assignment) for every function / dimension combination.

Redirect to a .tex file and \input it in your report:
    python generate_tables.py > tables.tex
"""

import pickle
import os
import numpy as np
from test_functions import ZakharovFunction, DixonPriceFunction, LevyFunction

FUNC_CLASSES = {
    'Zakharov':    ZakharovFunction,
    'Dixon-Price': DixonPriceFunction,
    'Levy':        LevyFunction,
}

_ROOT = os.path.dirname(os.path.abspath(__file__))


def rate_est(f_history, f_star, tail=0.3):
    """
    Estimate experimental convergence order p from |f_k - f*|.
    Uses the last `tail` fraction of the error sequence.
    Returns a float or None if the sequence is too short / already converged.
    """
    err = np.abs(np.array(f_history) - f_star)
    mask = (err > 1e-15) & np.isfinite(err)
    if mask.sum() < 6:
        return None
    err = err[mask]
    start = max(0, int(len(err) * (1 - tail)))
    e = err[start:]
    if len(e) < 4:
        return None
    rates = np.log(e[1:] + 1e-300) / (np.log(e[:-1] + 1e-300) + 1e-300)
    rates = rates[(rates > 0) & (rates < 5)]
    return float(np.median(rates)) if len(rates) else None


def fmt_rate(r):
    return f'{r:.2f}' if r is not None else '--'


def print_table(fname, n, dim_results, f_star, max_iter):
    labels = [r'\(\bar{x}\)'] + [rf'rand$_{{{k+1}}}$' for k in range(5)]

    print(f'\n% {fname},  n = {n}')
    print(r'\begin{tabular}{lllllll}')
    print(r'\toprule')
    print(r'Start pt. & grad.\ norm & iters/max & success & rate (exp.) & time \\')
    print(r'\midrule')

    success_rows = []
    for lbl, res in zip(labels, dim_results):
        rate = rate_est(res['f_history'], f_star)
        suc  = 'yes' if res['success'] else 'no'
        if res['success']:
            success_rows.append((res, rate))
        print(
            f"{lbl} & {res['grad_norm']:.2e} & "
            f"{res['n_iter']}/{max_iter} & {suc} & "
            f"{fmt_rate(rate)} & {res['time']:.2f}s \\\\"
        )

    if success_rows:
        avg_gnorm = np.mean([r['grad_norm'] for r, _ in success_rows])
        avg_iter  = np.mean([r['n_iter']    for r, _ in success_rows])
        avg_time  = np.mean([r['time']       for r, _ in success_rows])
        valid_rates = [rt for _, rt in success_rows if rt is not None]
        rate_avg = fmt_rate(np.mean(valid_rates)) if valid_rates else '--'
        print(r'\midrule')
        print(
            f"Avg (succ.) & {avg_gnorm:.2e} & "
            f"{avg_iter:.0f}/{max_iter} & -- & "
            f"{rate_avg} & {avg_time:.2f}s \\\\"
        )

    print(r'\bottomrule')
    print(r'\end{tabular}')


if __name__ == '__main__':
    with open(os.path.join(_ROOT, 'results.pkl'), 'rb') as fh:
        data = pickle.load(fh)

    results    = data['results']
    dimensions = data['dimensions']
    max_iter   = data.get('max_iter', data.get('nm_params', {}).get('max_iter', 50000))

    for fname, FuncClass in FUNC_CLASSES.items():
        print(f'\n\n%%% ============ {fname} ============')
        for n in dimensions:
            func = FuncClass(n)
            print_table(fname, n, results[fname][n], func.f_star, max_iter)
