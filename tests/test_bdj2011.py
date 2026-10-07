"""Regression tests on the one-factor Boogert and de Jong (2011) storage"""

import numpy as np
import pytest
from pytest import approx

from mettis import (
    FORWARD_CURVE,
    Factor,
    PriceModel,
    Storage,
    bdj_factors,
    intrinsic_value,
    lower_bound,
    solve_grid,
    solve_tt,
    upper_bound,
)

N_PATHS = 10_000


@pytest.fixture(scope="module")
def problem():
    storage = Storage()
    model = PriceModel(bdj_factors(1))
    grids = model.grids(81)
    grid_solution = solve_grid(model, storage, grids)
    tt_solution = solve_tt(model, storage, grids, eps=1e-6)
    return model, storage, grid_solution, tt_solution


def test_intrinsic_value():
    assert intrinsic_value(Storage(), FORWARD_CURVE) == approx(306.20, abs=1e-6)


def test_zero_volatility_gives_intrinsic_value():
    storage = Storage()
    model = PriceModel([Factor(kappa=0.12, sigma=1e-9)])
    solution = solve_grid(model, storage, model.grids(5), keep=False)
    assert solution.value == approx(intrinsic_value(storage, FORWARD_CURVE), abs=1e-4)


def test_grid_value(problem):
    _, _, grid_solution, _ = problem
    assert grid_solution.value == approx(656.187, abs=1e-3)


def test_tensor_train_value(problem):
    _, _, grid_solution, tt_solution = problem
    assert tt_solution.value == approx(grid_solution.value, abs=0.05)
    assert tt_solution.max_rank <= 30


def test_tensor_train_policy_matches_exact_policy(problem):
    model, storage, grid_solution, tt_solution = problem
    tt_policy, _ = lower_bound(model, storage, tt_solution, N_PATHS, seed=1)
    exact_policy, _ = lower_bound(model, storage, grid_solution, N_PATHS, seed=1)
    assert tt_policy == approx(exact_policy, abs=0.05)


def test_bounds_contain_exact_value(problem):
    model, storage, grid_solution, tt_solution = problem
    lower, lower_error = lower_bound(model, storage, tt_solution, N_PATHS, seed=1)
    upper, upper_error = upper_bound(model, storage, tt_solution, 2_000, seed=2)
    assert lower - 3 * lower_error <= grid_solution.value
    assert grid_solution.value <= upper + 3 * upper_error
    assert upper - grid_solution.value < 0.1
    assert np.isfinite([lower, upper]).all()
