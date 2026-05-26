# Nelder-Mead Optimizer

Implementation and experimental study of the **Nelder-Mead simplex method** for
unconstrained optimization, developed as the final project for the course
*Numerical Optimisation for Large Scale Problems*.

---

## Overview

The Nelder-Mead method (Nelder & Mead, 1965) is a derivative-free direct-search
algorithm for minimizing a scalar function of *n* variables. It iteratively
transforms a simplex of *n+1* vertices using four geometric operations:
**reflection**, **expansion**, **contraction**, and **shrink**.

This repository contains:

- A clean Python implementation of the algorithm (`nelder_mead.py`)
- Three *n*-dimensional benchmark functions (`test_functions.py`)
- Experiment runner with reproducible random seeds (`run_experiments.py`)
- Figure generator for convergence and top-view plots (`generate_figures.py`)

---

## Repository Structure

```
nelder-mead-optimizer/
├── nelder_mead.py        # Core Nelder-Mead implementation
├── test_functions.py     # Zakharov, Dixon-Price, and Levy benchmark functions
├── run_experiments.py    # Experiment runner (saves results.pkl)
├── generate_figures.py   # Figure generation (reads results.pkl)
├── requirements.txt      # Python dependencies
└── figures/              # Output directory for generated plots (gitignored)
```

---

## Benchmark Functions

All three functions are *n*-dimensional with a known global minimum at *f\* = 0*:

| Function     | Global minimiser         | Domain         | Suggested start         |
|--------------|--------------------------|----------------|-------------------------|
| Zakharov     | x\* = (0, …, 0)          | xᵢ ∈ [−5, 10] | x̄ = (0.5, …, 0.5)      |
| Dixon-Price  | x\*ᵢ = 2^(−(2ⁱ−2)/2ⁱ)   | xᵢ ∈ [−10, 10]| x̄ = (2, …, 2)           |
| Levy         | x\* = (1, …, 1)          | xᵢ ∈ [−10, 10]| x̄ = (−1, …, −1)         |

Each class also exposes an exact `gradient()` method used for diagnostics.

---

## Installation

```bash
git clone https://github.com/<your-username>/nelder-mead-optimizer.git
cd nelder-mead-optimizer
pip install -r requirements.txt
```

Python 3.10+ recommended.

---

## Usage

### 1 — Run all experiments

```bash
python run_experiments.py
```

This runs Nelder-Mead on all three functions in dimensions n ∈ {2, 10, 20, 50},
starting from the suggested point x̄ and 5 random perturbations around it.
Results are saved to `results.pkl`.

Configuration constants at the top of the file:

| Constant      | Default   | Description                              |
|---------------|-----------|------------------------------------------|
| `SEED`        | `314159`  | Master random seed (set to min student ID)|
| `DIMENSIONS`  | `[2, 10, 20, 50]` | Problem dimensions to test      |
| `N_RANDOM`    | `5`       | Number of random starting points         |
| `MAX_ITER`    | `50000`   | Maximum iterations per run               |

### 2 — Generate figures

```bash
python generate_figures.py
```

Requires `results.pkl` to exist (run step 1 first).
Saves the following PDFs to the project root:

- `topview_<function>.pdf` — contour map with NM paths (n = 2)
- `conv_<function>_n<dim>.pdf` — convergence curves per dimension
- `rates_<function>.pdf` — estimated local convergence rate bar charts

### 3 — Use the optimizer standalone

```python
import numpy as np
from nelder_mead import nelder_mead
from test_functions import LevyFunction

f       = LevyFunction(n=10)
simplex = np.random.randn(11, 10)   # (n+1) × n initial simplex

result = nelder_mead(f, simplex)

print(result['x_opt'])   # best point found
print(result['f_opt'])   # function value at x_opt
print(result['n_iter'])  # iterations taken
print(result['success']) # True if converged
```

---

## Algorithm Parameters

| Parameter | Default | Role                    |
|-----------|---------|-------------------------|
| `alpha`   | 1.0     | Reflection coefficient  |
| `gamma`   | 2.0     | Expansion coefficient   |
| `rho`     | 0.5     | Contraction coefficient |
| `sigma`   | 0.5     | Shrink coefficient      |
| `tol_f`   | 1e-8    | Convergence tolerance on f spread |
| `tol_x`   | 1e-8    | Convergence tolerance on simplex diameter |
| `max_iter`| 50000   | Maximum iterations      |

Convergence is declared when **both** `|f_worst − f_best| < tol_f` and
`max‖xᵢ − x_best‖ < tol_x` hold simultaneously.

---

## Return Value

`nelder_mead()` returns a dict:

```python
{
    'x_opt':     np.ndarray,   # best point found
    'f_opt':     float,        # f(x_opt)
    'n_iter':    int,          # iterations performed
    'success':   bool,         # True if converged within max_iter
    'time':      float,        # wall-clock time in seconds
    'history':   np.ndarray,   # best vertex at each iteration  (n_iter × n)
    'f_history': np.ndarray,   # f(best vertex) at each iteration
}
```

---

## Reference

Nelder, J. A. & Mead, R. (1965). A simplex method for function minimization.
*The Computer Journal*, 7(4), 308–313.

---

## Authors

Elia Zonta

Giuseppe Fontanella
