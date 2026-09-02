"""Vendored subset of TabularBench (serval-uni-lu/tabularbench, MIT License).

We cannot install the TabularBench package: it hard-pins torch==1.12.1 (wheels
only up to cp310), so it is uninstallable on our Python 3.12 / modern-torch stack.
Instead we vendor only the pieces we need, unmodified, with attribution.

Vendored so far:
- `relation_constraint.py` -- the constraint DSL (Feature, Constant, SafeDivision,
  operators) used to BUILD the CTU-13 Neris 360-constraint set. Self-contained
  (standard-library `typing` only).

To be vendored in M2 when trigger projection needs to CHECK data against the
constraints: `numpy_backend.py`, `backend.py`, `constraints.py`,
`constraints_checker.py`, `utils.py`, `utils/typing.py`.

Source: https://github.com/serval-uni-lu/tabularbench (commit bfb7541). See LICENSE
in this directory for the MIT terms.
"""
