"""Boogert and de Jong (2011) slow storage: grid and tensor train values, bounds

python benchmarks/bdj2011.py
python benchmarks/bdj2011.py --factors 2 --eps 1e-5
"""

from argparse import ArgumentParser
from time import time

from mettis import (
    FORWARD_CURVE,
    PriceModel,
    Storage,
    bdj_factors,
    intrinsic_value,
    lower_bound,
    solve_grid,
    solve_tt,
    upper_bound,
)


def parse_commandline():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--factors", type=int, default=1, choices=[1, 2])
    parser.add_argument("--nodes", type=int, default=81, help="grid nodes per factor")
    parser.add_argument("--eps", type=float, default=1e-6, help="TT accuracy")
    parser.add_argument("--lower-paths", type=int, default=40_000)
    parser.add_argument("--upper-paths", type=int, default=4_000)
    parser.add_argument(
        "--skip-exact-policy",
        action="store_true",
        help="do not simulate the exact grid policy on the lower bound paths",
    )
    return parser.parse_args()


def main():
    args = parse_commandline()
    storage = Storage()
    model = PriceModel(bdj_factors(args.factors))
    grids = model.grids(args.nodes)

    print(f"intrinsic value           {intrinsic_value(storage, FORWARD_CURVE):.2f}")

    start = time()
    grid_solution = solve_grid(model, storage, grids, keep=not args.skip_exact_policy)
    print(
        f"grid value                {grid_solution.value:.3f}"
        f"    {args.nodes}^{args.factors} nodes, {time() - start:.0f} s"
    )

    start = time()
    tt_solution = solve_tt(model, storage, grids, eps=args.eps)
    print(
        f"tensor train value        {tt_solution.value:.3f}"
        f"    eps {args.eps:g}, max rank {tt_solution.max_rank}, {time() - start:.0f} s"
    )

    start = time()
    mean, error = lower_bound(model, storage, tt_solution, args.lower_paths, seed=1)
    print(
        f"lower bound (TT policy)   {mean:.3f} +- {error:.3f}"
        f"    {args.lower_paths} paths, {time() - start:.0f} s"
    )

    if not args.skip_exact_policy:
        start = time()
        exact, error = lower_bound(
            model, storage, grid_solution, args.lower_paths, seed=1
        )
        print(
            f"exact policy, same paths  {exact:.3f} +- {error:.3f}"
            f"    difference {exact - mean:.3f}, {time() - start:.0f} s"
        )

    start = time()
    upper, error = upper_bound(model, storage, tt_solution, args.upper_paths, seed=2)
    print(
        f"upper bound               {upper:.3f} +- {error:.3f}"
        f"    {args.upper_paths} paths, {time() - start:.0f} s"
    )
    print(f"gap                       {100 * (upper - mean) / mean:.3f} %")


if __name__ == "__main__":
    main()
