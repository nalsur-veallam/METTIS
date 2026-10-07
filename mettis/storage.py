"""Gas storage contract"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Storage:
    """Gas storage with integer inventory levels

    An action a > 0 withdraws a units and a < 0 injects -a units, so the inventory
    moves from I to I - a and the cash flow of the day is a S - cost |a|.

    Attributes
    ----------
    capacity: int, default=100
        Maximum inventory
    start: int, default=0
        Inventory on the first day
    end: int, default=0
        Inventory required after the last day
    max_injection: int, default=1
        Units per day
    max_withdrawal: int, default=1
        Units per day
    cost: float, default=0.0
        Cost per unit injected or withdrawn
    discount: float, default=1.0
        One-day discount factor
    n_days: int, default=365
        Number of decision days
    """

    capacity: int = 100
    start: int = 0
    end: int = 0
    max_injection: int = 1
    max_withdrawal: int = 1
    cost: float = 0.0
    discount: float = 1.0
    n_days: int = 365

    def actions(self) -> range:
        return range(-self.max_injection, self.max_withdrawal + 1)

    def cash(self, a: int | np.ndarray, spot: float | np.ndarray) -> np.ndarray:
        return a * spot - self.cost * np.abs(a)

    def levels(self, t: int) -> tuple[int, int]:
        """Lowest and highest inventory at day t

        Only levels that are reachable from the start and from which the end level can
        still be reached are kept.
        """
        days_left = self.n_days - t
        low = max(
            0,
            self.end - self.max_injection * days_left,
            self.start - self.max_withdrawal * t,
        )
        high = min(
            self.capacity,
            self.end + self.max_withdrawal * days_left,
            self.start + self.max_injection * t,
        )
        return low, high
