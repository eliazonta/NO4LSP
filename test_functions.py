"""
test_functions.py
-----------------
Three test problems for unconstrained optimization, taken from:
  [1] Test Problems for Unconstrained Optimization,
      https://www.researchgate.net/publication/325314497

Functions implemented:
  1. Zakharov function
  2. Dixon-Price function
  3. Levy function

Each class stores:
  - n        : problem dimension
  - x_bar    : suggested starting point (from [1])
  - x_star   : known global minimiser
  - f_star   : known global minimum value

Authors: Elia Zonta, Giuseppe Fontanella, Fatima Ali
"""

import numpy as np


# ═══════════════════════════════════════════════════════════════════════════════
#  1. ZAKHAROV FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════
class ZakharovFunction:
    """
    Zakharov function (n-dimensional):

        f(x) = sum_{i=1}^{n} x_i^2
             + ( sum_{i=1}^{n} 0.5*i*x_i )^2
             + ( sum_{i=1}^{n} 0.5*i*x_i )^4

    Global minimum:  x^* = (0, ..., 0),   f^* = 0
    Domain:          x_i in [-5, 10]
    Suggested start: x_bar = (0.5, ..., 0.5)  [consistent with domain interior]

    Exact gradient:
        Let  s = sum_{i=1}^{n} 0.5*i*x_i.
        Then  df/dx_j = 2*x_j + (2*s + 4*s^3) * 0.5*j
    """
    name = "Zakharov"

    def __init__(self, n: int):
        self.n     = n
        self.x_bar  = 0.5 * np.ones(n)          # suggested starting point
        self.x_star = np.zeros(n)                # global minimiser
        self.f_star = 0.0                        # global minimum

    def __call__(self, x: np.ndarray) -> float:
        x = np.asarray(x, dtype=float)
        i    = np.arange(1, self.n + 1, dtype=float)
        s    = np.dot(0.5 * i, x)
        return float(np.dot(x, x) + s**2 + s**4)

    def gradient(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        i    = np.arange(1, self.n + 1, dtype=float)
        s    = np.dot(0.5 * i, x)
        return 2.0 * x + (2.0 * s + 4.0 * s**3) * (0.5 * i)


# ═══════════════════════════════════════════════════════════════════════════════
#  2. DIXON-PRICE FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════
class DixonPriceFunction:
    """
    Dixon-Price function (n-dimensional):

        f(x) = (x_1 - 1)^2
             + sum_{i=2}^{n} i * (2*x_i^2 - x_{i-1})^2

    Global minimum:  x^*_i = 2^{-(2^i - 2)/2^i},  i = 1,...,n;   f^* = 0
    Domain:          x_i in [-10, 10]
    Suggested start: x_bar = (2, ..., 2)

    Exact gradient (zero-based indexing used internally):
        df/dx_1 = 2*(x_1 - 1) - 2*(2*(2*x_2^2 - x_1))         [if n>1]
        df/dx_j = 4*j*(2*x_j^2 - x_{j-1})*2*x_j
                - 2*(j+1)*(2*x_{j+1}^2 - x_j)                  [1 < j < n]
        df/dx_n = 4*n*(2*x_n^2 - x_{n-1})*2*x_n               [last term]
    """
    name = "Dixon-Price"

    def __init__(self, n: int):
        self.n      = n
        self.x_bar  = 2.0 * np.ones(n)
        i_arr       = np.arange(1, n + 1, dtype=float)
        self.x_star = 2.0 ** (-(2.0**i_arr - 2.0) / 2.0**i_arr)
        self.f_star = 0.0

    def __call__(self, x: np.ndarray) -> float:
        x   = np.asarray(x, dtype=float)
        val = (x[0] - 1.0) ** 2
        if self.n > 1:
            i_arr = np.arange(2, self.n + 1, dtype=float)
            val  += np.sum(i_arr * (2.0 * x[1:]**2 - x[:-1])**2)
        return float(val)

    def gradient(self, x: np.ndarray) -> np.ndarray:
        x    = np.asarray(x, dtype=float)
        grad = np.zeros(self.n)
        grad[0] = 2.0 * (x[0] - 1.0)
        if self.n > 1:
            i_arr = np.arange(2, self.n + 1, dtype=float)       # shape (n-1,)
            inner = 2.0 * x[1:]**2 - x[:-1]                     # shape (n-1,)
            # contribution to x[1:] from terms i*(2x_i^2 - x_{i-1})^2
            grad[1:] += i_arr * 2.0 * inner * 4.0 * x[1:]
            # contribution to x[:-1] from those same terms (chain rule on x_{i-1})
            grad[:-1] -= i_arr * 2.0 * inner
        return grad


# ═══════════════════════════════════════════════════════════════════════════════
#  3. LEVY FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════
class LevyFunction:
    """
    Levy function (n-dimensional):

        w_i  = 1 + (x_i - 1)/4,   i = 1,...,n

        f(x) = sin^2(pi*w_1)
             + sum_{i=1}^{n-1} (w_i - 1)^2 * [1 + 10*sin^2(pi*w_{i+1})]
             + (w_n - 1)^2 * [1 + sin^2(2*pi*w_n)]

    Global minimum:  x^* = (1, ..., 1),   f^* = 0
    Domain:          x_i in [-10, 10]
    Suggested start: x_bar = (-1, ..., -1)

    Exact gradient: see implementation below (chain rule through w_i = 1+(x_i-1)/4,
    so dw_i/dx_i = 1/4 for all i).
    """
    name = "Levy"

    def __init__(self, n: int):
        self.n      = n
        self.x_bar  = -1.0 * np.ones(n)
        self.x_star = np.ones(n)
        self.f_star = 0.0

    # ----------------------------------------------------------------- helpers
    @staticmethod
    def _w(x: np.ndarray) -> np.ndarray:
        return 1.0 + (x - 1.0) / 4.0

    def __call__(self, x: np.ndarray) -> float:
        x   = np.asarray(x, dtype=float)
        w   = self._w(x)
        t1  = np.sin(np.pi * w[0])**2
        t3  = (w[-1] - 1.0)**2 * (1.0 + np.sin(2.0 * np.pi * w[-1])**2)
        val = t1 + t3
        if self.n > 1:
            val += np.sum(
                (w[:-1] - 1.0)**2 * (1.0 + 10.0 * np.sin(np.pi * w[1:])**2)
            )
        return float(val)

    def gradient(self, x: np.ndarray) -> np.ndarray:
        x    = np.asarray(x, dtype=float)
        w    = self._w(x)
        dw   = 0.25           # dw_i / dx_i = 1/4 for all i
        grad = np.zeros(self.n)

        # --- term 1: sin^2(pi*w_1) ---
        grad[0] += 2.0 * np.sin(np.pi * w[0]) * np.cos(np.pi * w[0]) * np.pi * dw

        # --- term 2: sum_{i=0}^{n-2} (w_i-1)^2 * [1 + 10*sin^2(pi*w_{i+1})] ---
        if self.n > 1:
            A = (w[:-1] - 1.0)**2                                  # shape (n-1,)
            B = 1.0 + 10.0 * np.sin(np.pi * w[1:])**2             # shape (n-1,)
            # d/dx_i  (i = 0..n-2)
            grad[:-1] += 2.0 * (w[:-1] - 1.0) * B * dw
            # d/dx_{i+1} (i = 0..n-2)
            grad[1:]  += A * 20.0 * np.sin(np.pi * w[1:]) * np.cos(np.pi * w[1:]) * np.pi * dw

        # --- term 3: (w_n-1)^2 * [1 + sin^2(2*pi*w_n)] ---
        wn    = w[-1]
        s2    = np.sin(2.0 * np.pi * wn)
        grad[-1] += (
            2.0 * (wn - 1.0) * (1.0 + s2**2)
            + (wn - 1.0)**2 * 2.0 * s2 * np.cos(2.0 * np.pi * wn) * 2.0 * np.pi
        ) * dw

        return grad


# ═══════════════════════════════════════════════════════════════════════════════
#  Registry
# ═══════════════════════════════════════════════════════════════════════════════
ALL_FUNCTIONS = [ZakharovFunction, DixonPriceFunction, LevyFunction]


# ─── quick self-test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    np.random.seed(0)
    for FuncClass in ALL_FUNCTIONS:
        f   = FuncClass(n=5)
        x0  = np.random.randn(5) * 0.1
        val = f(x0)
        grd = f.gradient(x0)
        # finite-difference check
        h   = 1e-6
        gfd = np.zeros(5)
        for j in range(5):
            ej      = np.zeros(5); ej[j] = h
            gfd[j]  = (f(x0 + ej) - f(x0 - ej)) / (2 * h)
        err = np.linalg.norm(grd - gfd)
        print(f"{FuncClass.name:15s}  f(x0)={val:.4e}  grad_err={err:.2e}")
