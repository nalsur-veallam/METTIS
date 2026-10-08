"""Lower and upper bounds on the storage value from a solution"""

import numpy as np

from mettis.prices import PriceModel
from mettis.solvers import Solution, as_network
from mettis.storage import Network, Storage


def mean_and_error(samples: np.ndarray) -> tuple[float, float]:
    """Mean and standard error for antithetic paths"""
    half = len(samples) // 2
    pairs = 0.5 * (samples[:half] + samples[half:])
    return float(samples.mean()), float(pairs.std(ddof=1) / np.sqrt(half))


def _cash_of_actions(network: Network, spots: list, n_paths: int) -> np.ndarray:
    """Cash flow of every joint action on every path, shape (n_actions, n_paths)"""
    return np.stack(
        [
            np.broadcast_to(network.cash(i, spots), (n_paths,))
            for i in range(len(network.actions))
        ]
    )


def lower_bound(
    model: PriceModel,
    asset: Storage | Network,
    solution: Solution,
    n_paths: int,
    seed: int,
) -> tuple[float, float]:
    """Value of the policy defined by the solution, estimated on simulated paths

    Every day the policy takes the joint action that maximizes the cash flow plus
    C(t, I - a, x). The value of any admissible policy is a lower bound on the value.

    Returns
    -------
    tuple[float, float]
        Mean and standard error
    """
    network = as_network(asset)
    x, spot = model.simulate(n_paths, seed)
    paths = np.arange(n_paths)
    level = np.tile([s.start for s in network.storages], (n_paths, 1))
    total = np.zeros(n_paths)
    discount = 1.0

    for t in range(network.n_days):
        lows = np.array([low for low, _ in network.levels(t + 1)])
        highs = np.array([high for _, high in network.levels(t + 1)])
        continuation = solution.continuation_at(t, x[t])
        cash = _cash_of_actions(network, list(spot[t].T), n_paths)

        best = np.full(n_paths, -np.inf)
        best_action = np.zeros(n_paths, dtype=int)
        for i, action in enumerate(network.actions):
            moved = level - action
            allowed = np.all((moved >= lows) & (moved <= highs), axis=1)
            index = tuple(np.clip(moved - lows, 0, highs - lows).T)
            candidate = cash[i] + continuation[(paths,) + index]
            candidate = np.where(allowed, candidate, -np.inf)
            better = candidate > best
            best = np.where(better, candidate, best)
            best_action = np.where(better, i, best_action)

        total += discount * cash[best_action, paths]
        level = level - network.actions[best_action]
        discount *= network.discount

    return mean_and_error(total)


def upper_bound(
    model: PriceModel,
    asset: Storage | Network,
    solution: Solution,
    n_paths: int,
    seed: int,
) -> tuple[float, float]:
    """Information relaxation bound (Brown, Smith and Sun, 2010)

    On every path the problem is solved with full knowledge of the future prices, and
    moving to inventories I' at day t is charged the penalty
    discount V(t + 1, I', x(t + 1)) - C(t, I', x(t)). The penalty has zero conditional
    mean when C is the conditional expectation of the discounted V, and the average
    over paths is then an upper bound on the value.

    Returns
    -------
    tuple[float, float]
        Mean and standard error
    """
    network = as_network(asset)
    n = network.n_storages
    x, spot = model.simulate(n_paths, seed)
    path_values = np.zeros((n_paths,) + (1,) * n)

    for t in reversed(range(network.n_days)):
        levels = [np.arange(low, high + 1) for low, high in network.levels(t)]
        next_levels = network.levels(t + 1)

        # V(T) and C(T - 1) are both zero, so the last day has no penalty
        if t < network.n_days - 1:
            penalty = network.discount * solution.value_at(
                t + 1, x[t + 1]
            ) - solution.continuation_at(t, x[t])
            path_values = network.discount * path_values - penalty

        cash = _cash_of_actions(network, list(spot[t].T), n_paths)
        new_values = np.full((n_paths,) + tuple(len(lv) for lv in levels), -np.inf)
        for i, action in enumerate(network.actions):
            targets, sources = [], []
            for j in range(n):
                moved = levels[j] - action[j]
                low, high = next_levels[j]
                allowed = (moved >= low) & (moved <= high)
                targets.append(np.flatnonzero(allowed))
                sources.append(moved[allowed] - low)
            if any(len(target) == 0 for target in targets):
                continue
            target = (slice(None),) + np.ix_(*targets)
            source = (slice(None),) + np.ix_(*sources)
            gain = cash[i].reshape((n_paths,) + (1,) * n)
            new_values[target] = np.maximum(
                new_values[target], gain + path_values[source]
            )
        path_values = new_values

    return mean_and_error(path_values.reshape(n_paths, -1)[:, 0])
