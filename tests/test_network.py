"""Tests of the multi-hub extension: two hubs, storages at both, one link"""

import numpy as np
import pytest
from pytest import approx

from mettis import (
    FORWARD_CURVE,
    Factor,
    Link,
    Network,
    PriceModel,
    Storage,
    lower_bound,
    solve_grid,
    solve_tt,
    upper_bound,
)

N_DAYS = 20
N_NODES = 21


@pytest.fixture(scope="module")
def model():
    forward = np.stack([FORWARD_CURVE[:N_DAYS], FORWARD_CURVE[:N_DAYS] + 0.5])
    common = Factor(kappa=0.12, sigma=1.0 / np.sqrt(365), hub_loadings=(1.0, 1.0))
    spread = Factor(kappa=0.05, sigma=0.3 / np.sqrt(365), hub_loadings=(0.0, 1.0))
    return PriceModel([common, spread], forward)


@pytest.fixture(scope="module")
def grids(model):
    return model.grids(N_NODES)


def storage(capacity: int = 10) -> Storage:
    if capacity == 0:
        return Storage(capacity=0, max_injection=0, max_withdrawal=0, n_days=N_DAYS)
    return Storage(capacity=capacity, n_days=N_DAYS)


def value(model, grids, network) -> float:
    return solve_grid(model, network, grids, keep=False).value


def test_independent_storages_add_up(model, grids):
    joint = value(model, grids, Network([storage(), storage()], hubs=[0, 1]))
    first = value(model, grids, Network([storage()], hubs=[0]))
    second = value(model, grids, Network([storage()], hubs=[1]))
    assert first > 0 and second > 0
    assert joint == approx(first + second, abs=1e-8)


def test_link_adds_a_strip_of_spread_options(model, grids):
    link = Link(source=0, target=1, capacity=1)
    with_link = value(model, grids, Network([storage()], hubs=[0], links=[link]))
    alone = value(model, grids, Network([storage()], hubs=[0]))
    strip = value(model, grids, Network([storage(0)], hubs=[0], links=[link]))
    assert strip > 0
    assert with_link == approx(alone + strip, abs=1e-8)


def test_hub_capacity_couples_storage_and_transport(model, grids):
    link = Link(source=0, target=1, capacity=1)
    storages = [storage(), storage()]
    without_link = value(model, grids, Network(storages, hubs=[0, 1]))
    coupled = value(
        model, grids, Network(storages, [0, 1], links=[link], hub_capacity=[1, 1])
    )
    free = value(model, grids, Network(storages, [0, 1], links=[link]))
    assert without_link < coupled < free


@pytest.fixture(scope="module")
def coupled_problem(model, grids):
    network = Network(
        [storage(), storage()],
        hubs=[0, 1],
        links=[Link(source=0, target=1, capacity=1)],
        hub_capacity=[1, 1],
    )
    grid_solution = solve_grid(model, network, grids)
    tt_solution = solve_tt(model, network, grids, eps=1e-5)
    return network, grid_solution, tt_solution


def test_tensor_train_matches_grid_for_two_storages(coupled_problem):
    _, grid_solution, tt_solution = coupled_problem
    assert tt_solution.value == approx(grid_solution.value, rel=1e-3)


def test_bounds_contain_exact_value_for_two_storages(model, coupled_problem):
    network, grid_solution, tt_solution = coupled_problem
    lower, lower_error = lower_bound(model, network, tt_solution, 4_000, seed=1)
    exact, _ = lower_bound(model, network, grid_solution, 4_000, seed=1)
    upper, upper_error = upper_bound(model, network, tt_solution, 1_000, seed=2)
    assert lower == approx(exact, rel=1e-3)
    assert lower - 3 * lower_error <= grid_solution.value <= upper + 3 * upper_error
