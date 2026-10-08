"""Factor model for spot prices at one or several hubs

The default forward curve is the one of the Boogert and de Jong (2011) benchmark.
"""

from dataclasses import dataclass

import numpy as np

# Monthly forward prices (p/therm) from April 2009 to March 2010, flat within a month
MONTH_DAYS = [30, 31, 30, 31, 31, 30, 31, 30, 31, 31, 28, 31]
MONTH_PRICES = [26.0, 25.0, 23.0, 22.0, 21.5, 21.0, 20.8, 20.7, 22.0, 23.0, 24.0, 25.5]
FORWARD_CURVE = np.repeat(MONTH_PRICES, MONTH_DAYS)

# 1 February counted from 1 April, the day where the seasonal loading peaks
FEBRUARY_FIRST = 306


@dataclass(frozen=True)
class Factor:
    """Factor of the log spot prices

    Attributes
    ----------
    kappa: float
        Daily mean reversion speed, zero gives a random walk
    sigma: float
        Daily volatility
    seasonal: bool, default=False
        Multiply the loading by 0.5 cos(2 pi (t - 1 Feb) / 365.25)
    hub_loadings: tuple[float, ...], default=None
        Loading of the factor at each hub, None means 1 at every hub. A factor with
        loadings (0, 1) moves only the second hub, that is the spread between the hubs.
    """

    kappa: float
    sigma: float
    seasonal: bool = False
    hub_loadings: tuple[float, ...] | None = None


def bdj_factors(n_factors: int) -> list[Factor]:
    """Factors of the benchmark

    The long-term factor of the benchmark is a martingale independent of the other
    factors. With zero operating costs it does not change the value and is left out,
    so the three-factor case is solved with the short-term and winter-summer factors.
    """
    short_term = Factor(kappa=0.12, sigma=1.0 / np.sqrt(365))
    winter_summer = Factor(kappa=0.0, sigma=0.2 / np.sqrt(365), seasonal=True)
    return [short_term, winter_summer][:n_factors]


class PriceModel:
    """Spot prices at one or several hubs driven by independent factors

    x_k(t + 1) = (1 - kappa_k) x_k(t) + sigma_k eps_k(t + 1)
    ln S_h(t) = ln F_h(t) + sum_k lambda_hk(t) x_k(t) - Var[sum_k lambda_hk(t) x_k(t)] / 2

    The variance term makes E[S_h(t)] = F_h(t).

    Attributes
    ----------
    factors: list[Factor]
        Factors of the model
    forward: np.ndarray
        Forward curves F_h(t) of shape (n_hubs, n_days)
    """

    def __init__(
        self, factors: list[Factor], forward: np.ndarray = FORWARD_CURVE
    ) -> None:
        self.factors = factors
        self.forward = np.atleast_2d(np.asarray(forward, dtype=float))
        self.n_hubs, self.n_days = self.forward.shape
        self.n_factors = len(factors)
        for factor in factors:
            if factor.hub_loadings is not None:
                assert len(factor.hub_loadings) == self.n_hubs

    def loading(self, k: int, t: int, hub: int = 0) -> float:
        factor = self.factors[k]
        value = 1.0 if factor.hub_loadings is None else factor.hub_loadings[hub]
        if factor.seasonal:
            value *= 0.5 * np.cos(2 * np.pi * (t - FEBRUARY_FIRST) / 365.25)
        return value

    def variance(self, k: int, t: int) -> float:
        """Variance of factor k at day t, the factor starts at zero"""
        a = 1.0 - self.factors[k].kappa
        sigma = self.factors[k].sigma
        if a == 1.0:
            return sigma**2 * t
        return sigma**2 * (1.0 - a ** (2 * t)) / (1.0 - a**2)

    def log_shift(self, t: int, hub: int = 0) -> float:
        return -0.5 * sum(
            self.loading(k, t, hub) ** 2 * self.variance(k, t)
            for k in range(self.n_factors)
        )

    def spot(self, t: int, x: np.ndarray, hub: int = 0) -> np.ndarray:
        """Spot price at day t for factor values x of shape (n_points, n_factors)"""
        z = sum(self.loading(k, t, hub) * x[:, k] for k in range(self.n_factors))
        return self.forward[hub, t] * np.exp(self.log_shift(t, hub) + z)

    def spot_on_grid(
        self, t: int, grids: list[np.ndarray], hub: int = 0
    ) -> tuple[float, list[np.ndarray]]:
        """Spot price on a grid as S[i_1, ..., i_d] = base prod_k parts[k][i_k]"""
        base = self.forward[hub, t] * np.exp(self.log_shift(t, hub))
        parts = [np.exp(self.loading(k, t, hub) * g) for k, g in enumerate(grids)]
        return base, parts

    def grids(self, n_nodes: int, n_std: float = 5.0) -> list[np.ndarray]:
        """Uniform grids over n_std standard deviations of each factor at the horizon

        Use an odd number of nodes, so that zero is a grid node.
        """
        return [
            np.linspace(-1.0, 1.0, n_nodes)
            * n_std
            * np.sqrt(self.variance(k, self.n_days))
            for k in range(self.n_factors)
        ]

    def simulate(self, n_paths: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
        """Simulate antithetic paths, the second half of the paths mirrors the first

        Returns
        -------
        x: np.ndarray
            Factor values of shape (n_days, n_paths, n_factors)
        spot: np.ndarray
            Spot prices of shape (n_days, n_paths, n_hubs)
        """
        rng = np.random.default_rng(seed)
        shocks = rng.standard_normal((self.n_days - 1, n_paths // 2, self.n_factors))
        shocks = np.concatenate([shocks, -shocks], axis=1)

        a = np.array([1.0 - factor.kappa for factor in self.factors])
        sigma = np.array([factor.sigma for factor in self.factors])
        x = np.zeros((self.n_days, n_paths, self.n_factors))
        for t in range(1, self.n_days):
            x[t] = a * x[t - 1] + sigma * shocks[t - 1]

        spot = np.stack(
            [
                np.stack([self.spot(t, x[t], h) for h in range(self.n_hubs)], axis=-1)
                for t in range(self.n_days)
            ]
        )
        return x, spot
