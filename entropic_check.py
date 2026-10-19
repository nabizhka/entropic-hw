"""Numerical checks for the entropic risk measure, rho(L) = (1/gamma) ln E[exp(gamma L)] (MF731 HW 7).

Each proved property is tested on simulated losses, independent of the proof:
  - cash-additivity rho(L - l) = rho(L) - l and monotonicity (L1 <= L2 => rho(L1) <= rho(L2))
  - convexity rho(lam L1 + (1 - lam) L2) <= lam rho(L1) + (1 - lam) rho(L2), with the gap
  - rho(lam Z) = gamma lam^2 / 2 for Z ~ N(0, 1), by Gauss-Hermite quadrature and Monte Carlo
Writes generated/checks.tex and generated/convexity.tex.
"""
import os

import numpy as np
from scipy.special import logsumexp

ROOT = os.path.join(os.path.dirname(__file__), "..")
rng = np.random.default_rng(731)
N = 2_000_000


def rho(L, g):
    """Sample entropic risk; logsumexp keeps exp(g L) from overflowing."""
    return (logsumexp(g * L) - np.log(len(L))) / g


# ---------------------------------------------------------------- Q1, Q2 on simulated pairs
# three loss models with finite exponential moments, deliberately non-normal
Z = rng.standard_normal(N)
U = rng.random(N)
pairs = {
    "normal / uniform": (Z, 3 * U - 1),
    "dependent: $L_2 = L_1^2/2 - 1$": (Z, Z**2 / 2 - 1),
    "Bernoulli(0.05)$\\times$10 / $-$Exp(1)": (10.0 * (U < 0.05), -rng.exponential(1.0, N)),
}
g = 0.4               # gamma < 1 keeps E exp(g Z^2 / 2) finite for the second pair
conv_rows, worst_gap = [], np.inf
for name, (L1, L2) in pairs.items():
    r1, r2 = rho(L1, g), rho(L2, g)
    # cash-additivity: exact on the sample (the shift factors out of logsumexp)
    for ell in (-3.0, 0.5, 7.0):
        assert abs(rho(L1 - ell, g) - (r1 - ell)) < 1e-10
    # monotonicity: L1 <= max(L1, L2) pathwise
    assert rho(L1, g) <= rho(np.maximum(L1, L2), g) + 1e-12
    for lam in (0.25, 0.5, 0.75):
        lhs = rho(lam * L1 + (1 - lam) * L2, g)
        rhs = lam * r1 + (1 - lam) * r2
        worst_gap = min(worst_gap, rhs - lhs)
        conv_rows.append((name, lam, lhs, rhs))
assert worst_gap >= -1e-12  # Hoelder holds on the empirical measure too, so exactly

# ---------------------------------------------------------------- Q3: rho(lam Z), Z ~ N(0, 1)
x, wq = np.polynomial.hermite_e.hermegauss(80)   # probabilists' Hermite: weight exp(-x^2/2)
wq = wq / wq.sum()
rows = []
for gam in (0.5, 1.0, 2.0):
    for lam in (0.5, 1.0, 2.0, 4.0):
        exact = gam * lam**2 / 2
        quad = np.log(np.sum(wq * np.exp(gam * lam * x))) / gam
        mc = rho(lam * Z, gam)
        homog = lam * gam / 2                     # what positive homogeneity would require
        assert abs(quad - exact) < 1e-10
        rows.append((gam, lam, exact, quad, mc, homog))
# subadditivity also fails: rho(Z + Z) = 2 gamma > gamma = rho(Z) + rho(Z)
assert rho(2 * Z, 1.0) > 2 * rho(Z, 1.0)

os.makedirs(os.path.join(ROOT, "generated"), exist_ok=True)
with open(os.path.join(ROOT, "generated", "checks.tex"), "w") as f:
    f.write("\\begin{tabular}{rrrrrr}\n\\toprule\n")
    f.write("$\\gamma$ & $\\lambda$ & $\\gamma\\lambda^2/2$ & Quadrature & Monte Carlo"
            " & $\\lambda\\rho(Z)=\\lambda\\gamma/2$ \\\\\n\\midrule\n")
    for gam, lam, ex, qu, mc, ho in rows:
        f.write(f"{gam:g} & {lam:g} & {ex:.4f} & {qu:.4f} & {mc:.4f} & {ho:.4f} \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")
with open(os.path.join(ROOT, "generated", "convexity.tex"), "w") as f:
    f.write("\\begin{tabular}{lrrr}\n\\toprule\n")
    f.write("Pair $(L_1, L_2)$ & $\\lambda$ & $\\rho(\\lambda L_1+(1-\\lambda)L_2)$"
            " & $\\lambda\\rho(L_1)+(1-\\lambda)\\rho(L_2)$ \\\\\n\\midrule\n")
    for name, lam, lhs, rhs in conv_rows:
        f.write(f"{name} & {lam:g} & {lhs:.4f} & {rhs:.4f} \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")

for row in rows:
    print("gamma %g lam %g exact %.4f quad %.4f mc %.4f homog %.4f" % row)
for name, lam, lhs, rhs in conv_rows:
    print(f"{name:45s} lam {lam:.2f}  lhs {lhs:8.4f} <= rhs {rhs:8.4f}")
print("smallest convexity gap", worst_gap)
