"""Backward induction for storage values on a full grid and in tensor train format

The state is the inventory of every storage and the factor values. Both solvers keep
only the inventory levels Network.levels(t) at day t, so the required end levels are
part of the state space and need no penalty. Arrays and tensor trains have the
inventories as their first indices, each counted from the lowest level of its day,
followed by the factor grid indices.
"""

from functools import reduce

import numpy as np
import teneva

from mettis.operators import cubic_stencil, transition_matrices
from mettis.prices import PriceModel
from mettis.storage import Network, Storage


def as_network(asset: Storage | Network) -> Network:
    return Network.single(asset) if isinstance(asset, Storage) else asset


def bellman(
    network: Network, t: int, spots: list, continuation: np.ndarray
) -> np.ndarray:
    """V(t, I, x) = max over joint actions of {cash + C(t, I - a, x)} for all levels I

    Parameters
    ----------
    network: Network
        Storages and links
    t: int
        Day
    spots: list
        Spot prices at each hub, scalars or arrays broadcastable to the factor shape
    continuation: np.ndarray
        C(t, ., x) with the inventories of day t + 1 as first indices

    Returns
    -------
    np.ndarray
        V(t, ., x) with the inventories of day t as first indices
    """
    n = network.n_storages
    levels = [np.arange(low, high + 1) for low, high in network.levels(t)]
    next_levels = network.levels(t + 1)

    shape = tuple(len(level) for level in levels) + continuation.shape[n:]
    values = np.full(shape, -np.inf)
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
        target, source = np.ix_(*targets), np.ix_(*sources)
        candidate = network.cash(i, spots) + continuation[source]
        values[target] = np.maximum(values[target], candidate)
    return values


def intrinsic_value(asset: Storage | Network, prices: np.ndarray) -> float:
    """Value of the best fixed schedule for deterministic prices of shape (n_hubs, n_days)"""
    network = as_network(asset)
    prices = np.atleast_2d(prices)
    values = np.zeros((1,) * network.n_storages)
    for t in reversed(range(network.n_days)):
        values = bellman(network, t, list(prices[:, t]), network.discount * values)
    return float(values.ravel()[0])


class Solution:
    """Value function of a storage problem

    For every day t the continuation values C(t, I, x) and the values V(t, I, x) are
    kept as tensor trains on the grid. The first indices are the inventories, counted
    from the lowest levels of day t + 1 for C and of day t for V.

    Attributes
    ----------
    value: float
        V(0, start, 0)
    grids: list[np.ndarray]
        Factor grids
    n_storages: int
        Number of inventory indices in front of the factor indices
    continuation: dict[int, list[np.ndarray]]
        Tensor train of C(t, ., .) for each day t
    values: dict[int, list[np.ndarray]]
        Tensor train of V(t, ., .) for each day t, empty for the grid solver
    ranks: list[list[int]]
        Ranks of V(t, ., .) for each day, empty for the grid solver
    """

    def __init__(
        self,
        value: float,
        grids: list[np.ndarray],
        n_storages: int,
        continuation: dict,
        values: dict | None = None,
        ranks: list | None = None,
    ) -> None:
        self.value = value
        self.grids = grids
        self.n_storages = n_storages
        self.continuation = continuation
        self.values = values if values is not None else {}
        self.ranks = ranks if ranks is not None else []

    @property
    def max_rank(self) -> int:
        return max(max(r) for r in self.ranks)

    def continuation_at(self, t: int, x: np.ndarray) -> np.ndarray:
        """C(t, ., x) for factor values x of shape (n_points, n_factors)"""
        return tt_at_points(self.continuation[t], self.grids, x, self.n_storages)

    def value_at(self, t: int, x: np.ndarray) -> np.ndarray:
        """V(t, ., x) for factor values x of shape (n_points, n_factors)"""
        return tt_at_points(self.values[t], self.grids, x, self.n_storages)


def spots_on_grid(model: PriceModel, t: int, grids: list[np.ndarray]) -> list:
    """Spot prices at every hub on the full factor grid"""
    spots = []
    for hub in range(model.n_hubs):
        base, parts = model.spot_on_grid(t, grids, hub)
        spots.append(base * reduce(np.multiply, np.ix_(*parts)))
    return spots


def solve_grid(
    model: PriceModel,
    asset: Storage | Network,
    grids: list[np.ndarray],
    keep: bool = True,
) -> Solution:
    """Exact backward induction on the full grid

    With keep=True the continuation values of every day are stored as tensor trains
    without compression, so that the exact policy can be simulated with the same code
    as the tensor train policy.
    """
    network = as_network(asset)
    n = network.n_storages
    transitions = transition_matrices(model, grids)
    values = np.zeros((1,) * n + tuple(len(grid) for grid in grids))
    continuation_trains = {}

    for t in reversed(range(network.n_days)):
        continuation = values
        for k, P in enumerate(transitions):
            continuation = np.tensordot(continuation, P, axes=([n + k], [1]))
            continuation = np.moveaxis(continuation, -1, n + k)
        continuation = network.discount * continuation
        if keep:
            continuation_trains[t] = teneva.svd(continuation, e=1e-12)
        values = bellman(network, t, spots_on_grid(model, t, grids), continuation)

    center = tuple(len(grid) // 2 for grid in grids)
    value = float(values[(0,) * n + center])
    return Solution(value, grids, n, continuation_trains)


def solve_tt(
    model: PriceModel,
    asset: Storage | Network,
    grids: list[np.ndarray],
    eps: float = 1e-5,
    max_rank: int = 60,
    n_sweeps: int = 8,
) -> Solution:
    """Backward induction with the value function in tensor train format

    The expectation C(t) = discount (I x ... x I x P_1 x ... x P_d) V(t + 1) is a
    Kronecker product, so it multiplies every factor core of V(t + 1) by its transition
    matrix and leaves the inventory cores alone. The maximum over actions is not linear
    in C(t), so V(t) is rebuilt by TT-cross from values of the Bellman right-hand side
    and rounded to relative accuracy eps.

    Parameters
    ----------
    model: PriceModel
        Price model
    asset: Storage | Network
        Storage contract or network of storages and links
    grids: list[np.ndarray]
        Factor grids
    eps: float, default=1e-5
        Relative accuracy of TT-cross and of the rounding after each day
    max_rank: int, default=60
        Rank limit of V(t, ., .)
    n_sweeps: int, default=8
        Maximum number of TT-cross sweeps per day
    """
    network = as_network(asset)
    n = network.n_storages
    transitions = transition_matrices(model, grids)

    # V(T) = 0 on the single admissible level of every storage
    train = [np.zeros((1, 1, 1))] + [np.ones((1, 1, 1))] * (n - 1)
    train += [np.ones((1, len(grid), 1)) for grid in grids]
    solution = Solution(np.nan, grids, n, {}, values={network.n_days: train})

    for t in reversed(range(network.n_days)):
        continuation = [network.discount * train[0]] + train[1:n]
        continuation += [
            np.einsum("ij,ajb->aib", P, core) for P, core in zip(transitions, train[n:])
        ]
        bellman_rhs = _bellman_rhs(model, network, grids, t, continuation)

        # Start from V(t + 1) mapped onto the inventory levels of day t
        start = []
        for core, (low, high), (next_low, next_high) in zip(
            train[:n], network.levels(t), network.levels(t + 1)
        ):
            index = np.clip(
                np.arange(low, high + 1) - next_low, 0, next_high - next_low
            )
            start.append(core[:, index, :])
        start += train[n:]

        train = teneva.cross(
            bellman_rhs, start, e=eps, nswp=n_sweeps, dr_min=1, dr_max=2
        )
        train = teneva.truncate(train, eps, r=max_rank)

        solution.continuation[t] = continuation
        solution.values[t] = train
        solution.ranks.append(teneva.ranks(train).tolist())

    center = [len(grid) // 2 for grid in grids]
    solution.value = float(tt_values(train, np.array([[0] * n + center]))[0])
    return solution


def _bellman_rhs(
    model: PriceModel,
    network: Network,
    grids: list[np.ndarray],
    t: int,
    continuation: list[np.ndarray],
):
    """Bellman right-hand side at day t as a function of grid multi-indices"""
    n = network.n_storages
    lows = np.array([low for low, _ in network.levels(t)])
    next_lows = np.array([low for low, _ in network.levels(t + 1)])
    next_highs = np.array([high for _, high in network.levels(t + 1)])
    spot_factors = [model.spot_on_grid(t, grids, hub) for hub in range(model.n_hubs)]

    def rhs(indices: np.ndarray) -> np.ndarray:
        indices = np.asarray(indices, dtype=int)
        levels = lows + indices[:, :n]
        spots = [
            base * np.prod([part[indices[:, n + k]] for k, part in enumerate(parts)], 0)
            for base, parts in spot_factors
        ]
        result = np.full(len(indices), -np.inf)
        for i, action in enumerate(network.actions):
            moved = levels - action
            allowed = np.all((moved >= next_lows) & (moved <= next_highs), axis=1)
            if allowed.any():
                next_indices = indices[allowed].copy()
                next_indices[:, :n] = moved[allowed] - next_lows
                cash = network.cash(i, [spot[allowed] for spot in spots])
                candidate = cash + tt_values(continuation, next_indices)
                result[allowed] = np.maximum(result[allowed], candidate)
        return result

    return rhs


def tt_values(cores: list[np.ndarray], indices: np.ndarray, chunk: int = 20_000):
    """Elements of a tensor train at integer multi-indices of shape (n_points, d)"""
    result = np.empty(len(indices))
    for i in range(0, len(indices), chunk):
        block = indices[i : i + chunk]
        product = cores[0][0, block[:, 0], :]
        for k in range(1, len(cores)):
            product = np.einsum("mr,rms->ms", product, cores[k][:, block[:, k], :])
        result[i : i + chunk] = product[:, 0]
    return result


def tt_at_points(
    cores: list[np.ndarray],
    grids: list[np.ndarray],
    x: np.ndarray,
    n_storages: int = 1,
    chunk: int = 2_000,
) -> np.ndarray:
    """Tensor train at continuous factor values, for all inventory levels

    The inventory cores are multiplied once. Cubic interpolation weights of each
    factor are applied to its core, then the factor cores are multiplied.

    Returns
    -------
    np.ndarray
        Values of shape (n_points, n_levels_1, ..., n_levels_K)
    """
    inventory = cores[0][0]
    for core in cores[1:n_storages]:
        inventory = np.tensordot(inventory, core, axes=([-1], [0]))

    result = []
    for i in range(0, len(x), chunk):
        product = None
        for k, grid in enumerate(grids):
            indices, weights = cubic_stencil(grid, x[i : i + chunk, k])
            core = cores[n_storages + k][:, indices, :]
            core = np.einsum("mj,rmjs->mrs", weights, core)
            product = core if product is None else product @ core
        result.append(np.einsum("...r,mr->m...", inventory, product[:, :, 0]))
    return np.concatenate(result)


def _tt_dot(a: list[np.ndarray], b: list[np.ndarray]) -> float:
    product = np.ones((1, 1))
    for core_a, core_b in zip(a, b):
        product = np.einsum("ab,aic,bid->cd", product, core_a, core_b, optimize=True)
    return float(product[0, 0])


def _tt_relative_difference(a: list[np.ndarray], b: list[np.ndarray]) -> float:
    """||a - b|| / ||b|| computed from inner products"""
    aa, ab, bb = _tt_dot(a, a), _tt_dot(a, b), _tt_dot(b, b)
    if bb == 0.0:
        return 0.0 if aa == 0.0 else np.inf
    return float(np.sqrt(max(aa - 2 * ab + bb, 0.0) / bb))


# teneva.cross calls teneva.accuracy after every sweep. The library version builds
# arrays of size r^4 n, several GB at ranks around 60, so we use the inner products.
teneva.accuracy = _tt_relative_difference
