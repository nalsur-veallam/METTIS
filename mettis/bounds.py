"""Lower and upper bounds on the storage value from a solution"""

import numpy as np

from mettis.prices import PriceModel
from mettis.solvers import Solution
from mettis.storage import Storage


def mean_and_error(samples: np.ndarray) -> tuple[float, float]:
    """Mean and standard error for antithetic paths"""
    half = len(samples) // 2
    pairs = 0.5 * (samples[:half] + samples[half:])
    return float(samples.mean()), float(pairs.std(ddof=1) / np.sqrt(half))


def lower_bound(
    model: PriceModel,
    storage: Storage,
    solution: Solution,
    n_paths: int,
    seed: int,
) -> tuple[float, float]:
    """Value of the policy defined by the solution, estimated on simulated paths

    Every day the policy takes the action that maximizes
    a S - cost |a| + C(t, I - a, x).
    The value of any admissible policy is a lower bound on the storage value.

    Returns
    -------
    tuple[float, float]
        Mean and standard error
    """
    x, spot = model.simulate(n_paths, seed)
    paths = np.arange(n_paths)
    level = np.full(n_paths, storage.start)
    total = np.zeros(n_paths)
    discount = 1.0

    for t in range(storage.n_days):
        next_low, next_high = storage.levels(t + 1)
        continuation = solution.continuation_at(t, x[t])

        best = np.full(n_paths, -np.inf)
        best_action = np.zeros(n_paths, dtype=int)
        for a in storage.actions():
            next_level = level - a
            allowed = (next_level >= next_low) & (next_level <= next_high)
            column = np.clip(next_level - next_low, 0, next_high - next_low)
            candidate = storage.cash(a, spot[t]) + continuation[paths, column]
            candidate = np.where(allowed, candidate, -np.inf)
            better = candidate > best
            best = np.where(better, candidate, best)
            best_action = np.where(better, a, best_action)

        total += discount * storage.cash(best_action, spot[t])
        level = level - best_action
        discount *= storage.discount

    return mean_and_error(total)


def upper_bound(
    model: PriceModel,
    storage: Storage,
    solution: Solution,
    n_paths: int,
    seed: int,
) -> tuple[float, float]:
    """Information relaxation bound (Brown, Smith and Sun, 2010)

    On every path the problem is solved with full knowledge of the future prices, and
    moving to level I' at day t is charged the penalty
    discount V(t + 1, I', x(t + 1)) - C(t, I', x(t)). The penalty has zero conditional
    mean when C is the conditional expectation of the discounted V, and the average
    over paths is then an upper bound on the storage value.

    Returns
    -------
    tuple[float, float]
        Mean and standard error
    """
    x, spot = model.simulate(n_paths, seed)
    path_values = np.zeros((n_paths, 1))

    for t in reversed(range(storage.n_days)):
        low, high = storage.levels(t)
        next_low, next_high = storage.levels(t + 1)

        # V(T) and C(T - 1) are both zero, so the last day has no penalty
        if t < storage.n_days - 1:
            penalty = storage.discount * solution.value_at(
                t + 1, x[t + 1]
            ) - solution.continuation_at(t, x[t])
            path_values = storage.discount * path_values - penalty

        levels = np.arange(low, high + 1)
        new_values = np.full((n_paths, len(levels)), -np.inf)
        for a in storage.actions():
            next_levels = levels - a
            allowed = (next_levels >= next_low) & (next_levels <= next_high)
            candidate = (
                storage.cash(a, spot[t])[:, None]
                + path_values[:, next_levels[allowed] - next_low]
            )
            new_values[:, allowed] = np.maximum(new_values[:, allowed], candidate)
        path_values = new_values

    return mean_and_error(path_values[:, 0])
