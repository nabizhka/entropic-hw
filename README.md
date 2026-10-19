# The entropic risk measure (MF731 HW 7)

rho(L) = (1/gamma) ln E[exp(gamma L)], gamma > 0.

1. Cash-additivity: the constant e^{-gamma l} leaves the expectation. Monotonicity: compose
   increasing maps.
2. Convexity: Hoelder with p = 1/lam, q = 1/(1 - lam) on e^{gamma lam L1}, e^{gamma (1-lam) L2}.
3. L ~ N(0,1): rho(lam L) = gamma lam^2 / 2 != lam gamma / 2 for lam != 1, so positive
   homogeneity fails (sub-additivity too: rho(2L) = 2 gamma > gamma). Convex, not coherent.

```bash
pip install -r requirements.txt
python src/entropic_check.py   # writes generated/checks.tex, generated/convexity.tex
latexmk -pdf main.tex
```
