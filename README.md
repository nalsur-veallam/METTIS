# METTIS

Multi-factor Exercise via Tensor-Train Inventory Solver.

Valuation of gas storage by backward induction on a grid and in tensor train format,
with a lower bound from the simulated policy and an upper bound from information
relaxation. The method is described in `../method_note/note_en.pdf`.

## Installation

```bash
pip install -e .            # numpy, teneva
pip install -e ".[test]"    # with pytest
```

## Usage

```python
from mettis import PriceModel, Storage, bdj_factors, solve_tt, lower_bound, upper_bound

storage = Storage()
model = PriceModel(bdj_factors(1))
grids = model.grids(81)

solution = solve_tt(model, storage, grids, eps=1e-6)
lower_bound(model, storage, solution, n_paths=40_000, seed=1)
upper_bound(model, storage, solution, n_paths=4_000, seed=2)
```

Benchmark of Boogert and de Jong (2011), slow storage:

```bash
python benchmarks/bdj2011.py                          # one factor, under a minute
python benchmarks/bdj2011.py --factors 2 --eps 1e-5   # two factors, about 15 minutes
```

| | one factor, eps 1e-6 | two factors, eps 1e-5 |
|---|---|---|
| intrinsic value | 306.20 | 306.20 |
| grid value, 81 nodes per factor | 656.187 | 685.201 |
| tensor train value (max rank) | 656.208 (23) | 687.638 (25) |
| lower bound, TT policy, 40000 paths | 655.618 +- 0.397 | 685.125 +- 0.542 |
| exact grid policy, same paths | 655.614 +- 0.397 | 685.143 +- 0.542 |
| upper bound, 4000 paths | 656.235 +- 0.002 | 685.387 +- 0.023 |

Full output is in `benchmarks/results/`. The tensor train policy and the exact policy
give the same value on the same paths, and the two bounds contain the grid value.
The raw tensor train value is biased upward by the rounding in each step and is not
used as an estimate. The three-factor benchmark has two non-trivial factors: the
long-term factor is a martingale and does not change the value when costs are zero.

## Tests

```bash
pytest tests
```

## Layout

| Module | Contents |
|---|---|
| `mettis/prices.py` | forward curve, factor model, grids, simulation |
| `mettis/operators.py` | cubic interpolation, one-day transition matrices |
| `mettis/storage.py` | storage contract: actions, cash flow, admissible levels |
| `mettis/solvers.py` | Bellman step, intrinsic value, grid and tensor train solvers |
| `mettis/bounds.py` | lower and upper bounds |
| `benchmarks/` | runs that reproduce the reported numbers |
| `tests/` | regression tests on the one-factor benchmark |

`mettis/solvers.py` replaces `teneva.accuracy`, which `teneva.cross` calls after every
sweep. The library version needs several GB of memory at ranks around 60; the
replacement computes the same relative difference from inner products.
