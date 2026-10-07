"""Interpolation on uniform grids and one-day transition matrices of the factors"""

import numpy as np

from mettis.prices import PriceModel


def cubic_stencil(grid: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Local cubic interpolation on a uniform grid

    f(y) is approximated by sum_j weights[:, j] f(grid[indices[:, j]]) over four
    neighbouring nodes. The two boundary cells use linear interpolation, and points
    outside the grid are moved to its ends.

    Returns
    -------
    indices: np.ndarray
        Grid indices of shape (len(y), 4)
    weights: np.ndarray
        Interpolation weights of shape (len(y), 4)
    """
    n_nodes, step = len(grid), grid[1] - grid[0]
    s = (np.clip(y, grid[0], grid[-1]) - grid[0]) / step
    i = np.clip(np.floor(s).astype(int), 0, n_nodes - 2)
    u = s - i

    indices = np.clip(np.stack([i - 1, i, i + 1, i + 2], axis=1), 0, n_nodes - 1)
    weights = np.stack(
        [
            -u * (u - 1) * (u - 2) / 6,
            (u + 1) * (u - 1) * (u - 2) / 2,
            -(u + 1) * u * (u - 2) / 2,
            (u + 1) * u * (u - 1) / 6,
        ],
        axis=1,
    )
    edge = (i == 0) | (i == n_nodes - 2)
    weights[edge] = np.stack(
        [np.zeros(edge.sum()), 1 - u[edge], u[edge], np.zeros(edge.sum())], axis=1
    )
    return indices, weights


def transition_matrix(
    grid: np.ndarray, a: float, sigma: float, n_quadrature: int = 24
) -> np.ndarray:
    """Matrix P with (P f)[i] close to E f(a grid[i] + sigma eps), eps ~ N(0, 1)

    Gauss-Hermite quadrature over the shock, cubic interpolation between grid nodes.
    """
    nodes, quadrature_weights = np.polynomial.hermite_e.hermegauss(n_quadrature)
    quadrature_weights = quadrature_weights / quadrature_weights.sum()

    n_nodes = len(grid)
    rows = np.repeat(np.arange(n_nodes), 4)
    P = np.zeros((n_nodes, n_nodes))
    for z, w in zip(nodes, quadrature_weights):
        indices, weights = cubic_stencil(grid, a * grid + sigma * z)
        np.add.at(P, (rows, indices.ravel()), w * weights.ravel())
    return P


def transition_matrices(model: PriceModel, grids: list[np.ndarray]) -> list[np.ndarray]:
    """One transition matrix per factor"""
    return [
        transition_matrix(grid, 1.0 - factor.kappa, factor.sigma)
        for grid, factor in zip(grids, model.factors)
    ]
