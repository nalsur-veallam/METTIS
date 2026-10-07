"""Backward induction for the storage value on a full grid and in tensor train format

Both solvers keep only the inventory levels Storage.levels(t) at day t. The required
end level is then part of the state space and needs no penalty.
"""

from functools import reduce

import numpy as np
import teneva

from mettis.operators import cubic_stencil, transition_matrices
from mettis.prices import PriceModel
from mettis.storage import Storage


def bellman(
    storage: Storage, t: int, spot: float | np.ndarray, continuation: np.ndarray
) -> np.ndarray:
    """V(t, I, x) = max_a {a S(x) - cost |a| + C(t, I - a, x)} for all levels I at day t

    Parameters
    ----------
    storage: Storage
        Storage contract
    t: int
        Day
    spot: float | np.ndarray
        Spot prices, broadcastable to continuation.shape[1:]
    continuation: np.ndarray
        Continuation values C(t, ., x), the first index is the inventory at day t + 1
        counted from storage.levels(t + 1)[0]

    Returns
    -------
    np.ndarray
        Values V(t, ., x), the first index counted from storage.levels(t)[0]
    """
    low, high = storage.levels(t)
    next_low, next_high = storage.levels(t + 1)
    levels = np.arange(low, high + 1)

    values = np.full((len(levels),) + continuation.shape[1:], -np.inf)
    for a in storage.actions():
        next_levels = levels - a
        allowed = (next_levels >= next_low) & (next_levels <= next_high)
        if allowed.any():
            candidate = (
                storage.cash(a, spot) + continuation[next_levels[allowed] - next_low]
            )
            values[allowed] = np.maximum(values[allowed], candidate)
    return values


def intrinsic_value(storage: Storage, prices: np.ndarray) -> float:
    """Value of the best fixed schedule for a deterministic price curve"""
    values = np.zeros(1)
    for t in reversed(range(storage.n_days)):
        values = bellman(storage, t, prices[t], storage.discount * values)
    return float(values[0])


class Solution:
    """Value function of a storage problem

    For every day t the continuation values C(t, I, x) and the values V(t, I, x) are
    kept as tensor trains on the grid. The first index is the inventory, counted from
    the lowest level of day t + 1 for C and of day t for V.

    Attributes
    ----------
    value: float
        V(0, start, 0)
    grids: list[np.ndarray]
        Factor grids
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
        continuation: dict,
        values: dict | None = None,
        ranks: list | None = None,
    ) -> None:
        self.value = value
        self.grids = grids
        self.continuation = continuation
        self.values = values if values is not None else {}
        self.ranks = ranks if ranks is not None else []

    @property
    def max_rank(self) -> int:
        return max(max(r) for r in self.ranks)

    def continuation_at(self, t: int, x: np.ndarray) -> np.ndarray:
        """C(t, ., x) for factor values x of shape (n_points, n_factors)"""
        return tt_at_points(self.continuation[t], self.grids, x)

    def value_at(self, t: int, x: np.ndarray) -> np.ndarray:
        """V(t, ., x) for factor values x of shape (n_points, n_factors)"""
        return tt_at_points(self.values[t], self.grids, x)


def solve_grid(
    model: PriceModel, storage: Storage, grids: list[np.ndarray], keep: bool = True
) -> Solution:
    """Exact backward induction on the full grid

    With keep=True the continuation values of every day are stored as tensor trains
    without compression, so that the exact policy can be simulated with the same code
    as the tensor train policy.
    """
    transitions = transition_matrices(model, grids)
    values = np.zeros((1,) + tuple(len(grid) for grid in grids))
    continuation_trains = {}

    for t in reversed(range(storage.n_days)):
        continuation = values
        for k, P in enumerate(transitions):
            continuation = np.tensordot(continuation, P, axes=([k + 1], [1]))
            continuation = np.moveaxis(continuation, -1, k + 1)
        continuation = storage.discount * continuation
        if keep:
            continuation_trains[t] = teneva.svd(continuation, e=1e-12)

        base, parts = model.spot_on_grid(t, grids)
        spot = base * reduce(np.multiply, np.ix_(*parts))
        values = bellman(storage, t, spot, continuation)

    value = float(values[(0,) + tuple(len(grid) // 2 for grid in grids)])
    return Solution(value, grids, continuation_trains)


def solve_tt(
    model: PriceModel,
    storage: Storage,
    grids: list[np.ndarray],
    eps: float = 1e-5,
    max_rank: int = 60,
    n_sweeps: int = 8,
) -> Solution:
    """Backward induction with the value function in tensor train format

    The expectation C(t) = discount (I x P_1 x ... x P_d) V(t + 1) is a Kronecker
    product, so it multiplies every factor core of V(t + 1) by its transition matrix.
    The maximum over actions is not linear in C(t), so V(t) is rebuilt by TT-cross
    from values of the Bellman right-hand side and rounded to relative accuracy eps.

    Parameters
    ----------
    model: PriceModel
        Price model
    storage: Storage
        Storage contract
    grids: list[np.ndarray]
        Factor grids
    eps: float, default=1e-5
        Relative accuracy of TT-cross and of the rounding after each day
    max_rank: int, default=60
        Rank limit of V(t, ., .)
    n_sweeps: int, default=8
        Maximum number of TT-cross sweeps per day
    """
    transitions = transition_matrices(model, grids)
    train = [np.zeros((1, 1, 1))] + [np.ones((1, len(grid), 1)) for grid in grids]
    solution = Solution(np.nan, grids, continuation={}, values={storage.n_days: train})

    for t in reversed(range(storage.n_days)):
        continuation = [storage.discount * train[0]] + [
            np.einsum("ij,ajb->aib", P, core) for P, core in zip(transitions, train[1:])
        ]
        bellman_rhs = _bellman_rhs(model, storage, grids, t, continuation)

        # Start from V(t + 1) mapped onto the inventory levels of day t
        low, high = storage.levels(t)
        next_low, next_high = storage.levels(t + 1)
        level_index = np.clip(
            np.arange(low, high + 1) - next_low, 0, next_high - next_low
        )
        start = [train[0][:, level_index, :]] + train[1:]

        train = teneva.cross(
            bellman_rhs, start, e=eps, nswp=n_sweeps, dr_min=1, dr_max=2
        )
        train = teneva.truncate(train, eps, r=max_rank)

        solution.continuation[t] = continuation
        solution.values[t] = train
        solution.ranks.append(teneva.ranks(train).tolist())

    center = [len(grid) // 2 for grid in grids]
    solution.value = float(tt_values(train, np.array([[0] + center]))[0])
    return solution


def _bellman_rhs(
    model: PriceModel,
    storage: Storage,
    grids: list[np.ndarray],
    t: int,
    continuation: list[np.ndarray],
):
    """Bellman right-hand side at day t as a function of grid multi-indices"""
    low, _ = storage.levels(t)
    next_low, next_high = storage.levels(t + 1)
    base, parts = model.spot_on_grid(t, grids)

    def rhs(indices: np.ndarray) -> np.ndarray:
        indices = np.asarray(indices, dtype=int)
        levels = low + indices[:, 0]
        spot = base * np.prod(
            [part[indices[:, k + 1]] for k, part in enumerate(parts)], axis=0
        )
        result = np.full(len(indices), -np.inf)
        for a in storage.actions():
            next_levels = levels - a
            allowed = (next_levels >= next_low) & (next_levels <= next_high)
            if allowed.any():
                next_indices = indices[allowed].copy()
                next_indices[:, 0] = next_levels[allowed] - next_low
                candidate = storage.cash(a, spot[allowed]) + tt_values(
                    continuation, next_indices
                )
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
    cores: list[np.ndarray], grids: list[np.ndarray], x: np.ndarray, chunk: int = 2_000
) -> np.ndarray:
    """Tensor train with the inventory as first index at continuous factor values

    Cubic interpolation weights of each factor are applied to its core, then the cores
    are multiplied.

    Returns
    -------
    np.ndarray
        Values of shape (n_points, n_levels)
    """
    result = []
    for i in range(0, len(x), chunk):
        product = None
        for k, grid in enumerate(grids):
            indices, weights = cubic_stencil(grid, x[i : i + chunk, k])
            core = np.einsum("mj,rmjs->mrs", weights, cores[k + 1][:, indices, :])
            product = core if product is None else product @ core
        result.append(product[:, :, 0] @ cores[0][0].T)
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
