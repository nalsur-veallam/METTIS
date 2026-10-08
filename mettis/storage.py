"""Gas storages, transport links between hubs and networks of both"""

from dataclasses import dataclass
from itertools import product

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


@dataclass(frozen=True)
class Link:
    """Transport capacity between two hubs

    A flow f > 0 buys f units at the source hub and sells them at the target hub,
    f < 0 transports in the other direction. The cash flow is
    f (S_target - S_source) - tariff |f|.

    Attributes
    ----------
    source: int
        Index of the source hub
    target: int
        Index of the target hub
    capacity: int
        Units per day in each direction
    tariff: float, default=0.0
        Cost per unit transported
    """

    source: int
    target: int
    capacity: int
    tariff: float = 0.0


class Network:
    """Storages at hubs connected by links

    Every storage trades at its hub. The net quantity sold at hub h on a day,
    sum of the withdrawals of storages at h plus the flows arriving at h minus the
    flows leaving h, can be limited by hub_capacity[h]. Such a limit makes storages
    and links compete for the hub, so the joint problem does not split into
    independent single-storage problems.

    Attributes
    ----------
    storages: list[Storage]
        Storages, all with the same number of days and discount factor
    hubs: list[int]
        Hub of each storage
    links: list[Link]
        Transport links
    hub_capacity: list[int | None]
        Limit on the net daily quantity sold or bought at each hub, None for no limit
    actions: np.ndarray
        Admissible joint actions of the storages, shape (n_actions, n_storages)
    flows: np.ndarray
        Link flows of the same joint actions, shape (n_actions, n_links)
    """

    def __init__(
        self,
        storages: list[Storage],
        hubs: list[int] | None = None,
        links: list[Link] | None = None,
        hub_capacity: list[int | None] | None = None,
    ) -> None:
        self.storages = storages
        self.hubs = hubs if hubs is not None else [0] * len(storages)
        self.links = links if links is not None else []
        n_hubs = 1 + max(
            self.hubs
            + [link.source for link in self.links]
            + [link.target for link in self.links]
        )
        self.hub_capacity = (
            hub_capacity if hub_capacity is not None else [None] * n_hubs
        )

        self.n_storages = len(storages)
        self.n_days = storages[0].n_days
        self.discount = storages[0].discount
        for storage in storages:
            assert storage.n_days == self.n_days and storage.discount == self.discount

        self.actions, self.flows = self._joint_actions()

    @staticmethod
    def single(storage: Storage) -> "Network":
        return Network([storage])

    def _joint_actions(self) -> tuple[np.ndarray, np.ndarray]:
        moves = [storage.actions() for storage in self.storages]
        flows = [range(-link.capacity, link.capacity + 1) for link in self.links]

        actions, link_flows = [], []
        for combination in product(*moves, *flows):
            a, f = combination[: self.n_storages], combination[self.n_storages :]
            if self._within_hub_capacity(a, f):
                actions.append(a)
                link_flows.append(f)
        n_actions = len(actions)
        actions = np.array(actions, dtype=int).reshape(n_actions, self.n_storages)
        link_flows = np.array(link_flows, dtype=int).reshape(n_actions, len(self.links))
        return actions, link_flows

    def _within_hub_capacity(self, a: tuple, f: tuple) -> bool:
        for h, capacity in enumerate(self.hub_capacity):
            if capacity is None:
                continue
            net = sum(a_j for a_j, hub in zip(a, self.hubs) if hub == h)
            net += sum(f_l for f_l, link in zip(f, self.links) if link.target == h)
            net -= sum(f_l for f_l, link in zip(f, self.links) if link.source == h)
            if abs(net) > capacity:
                return False
        return True

    def cash(self, i: int, spots: list) -> np.ndarray:
        """Cash flow of joint action i for spot prices spots[h] at each hub"""
        total = 0.0
        for j, (storage, hub) in enumerate(zip(self.storages, self.hubs)):
            total = total + storage.cash(self.actions[i, j], spots[hub])
        for k, link in enumerate(self.links):
            f = self.flows[i, k]
            if f != 0:
                total = total + f * (spots[link.target] - spots[link.source])
                total = total - link.tariff * abs(f)
        return total

    def levels(self, t: int) -> list[tuple[int, int]]:
        """Lowest and highest inventory of every storage at day t"""
        return [storage.levels(t) for storage in self.storages]
