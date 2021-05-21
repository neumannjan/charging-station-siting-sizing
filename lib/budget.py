import itertools
from typing import Dict, Iterable, List, Tuple, Union

import iteround
import numpy as np
import pandas as pd


class BudgetDist:
    def __getitem__(self, station_id: int) -> int:
        raise NotImplementedError

    def __setitem__(self, station_id: int, value: int):
        raise NotImplementedError

    def __iter__(self):
        raise NotImplementedError

    def copy(self):
        raise NotImplementedError

    def __add__(self, second):
        return ListBudgetDist([u + v for u, v in itertools.zip_longest(list(self), list(second), fillvalue=0)])

    def to_list(self, max_station: int) -> List[int]:
        raise NotImplementedError


class ListBudgetDist(BudgetDist):
    def __init__(self, values: Iterable[int]):
        if isinstance(values, list):
            self._values = list(values)
        else:
            self._values = [v for v in values]

    def __getitem__(self, station_id: int) -> int:
        return self._values[station_id]

    def __setitem__(self, station_id: int, value: int):
        self._values[station_id] = value

    def __iter__(self):
        return iter(self._values)

    def copy(self) -> 'ListBudgetDist':
        return ListBudgetDist(self._values.copy())

    def to_list(self, max_station: int) -> List[int]:
        return [*self._values, *([0] * (max_station + 1 - len(self._values)))]

    def __repr__(self):
        vals = ', '.join(["%02d" % v for v in iter(self)])
        return f"{self.__class__.__name__}([{vals}])"


class DictBudgetDist(BudgetDist):
    def __init__(self, values: Union[Dict[int, int], Iterable[Tuple[int, int]]]):
        if isinstance(values, dict):
            self._values = dict(values)
        else:
            self._values = {k: v for k, v in values}

    def __getitem__(self, station_id: int) -> int:
        return self._values.get(station_id, 0)

    def __setitem__(self, station_id: int, value: int):
        assert station_id in self._values.keys()

        self._values[station_id] = value

    def __iter__(self):
        if len(self._values) > 0:
            return iter(self.to_list(max(self._values.keys())))
        else:
            return iter(list())

    def copy(self) -> 'DictBudgetDist':
        return DictBudgetDist(self._values.copy())

    def to_list(self, max_station: int) -> List[int]:
        return [self._values.get(i, 0) for i in range(max_station + 1)]


BudgetDistLike = Union[List[int], BudgetDist]


def uniform_budget(traffic: pd.DataFrame, sum_budget: int) -> DictBudgetDist:
    station_ids = traffic['station'].unique()

    values = iteround.saferound(
        [sum_budget / len(station_ids) for _ in station_ids], places=0)
    return DictBudgetDist({i: int(val) for i, val in zip(station_ids, values)})


def weighed_budget(traffic: pd.DataFrame, sum_budget: int) -> DictBudgetDist:
    station_ids = traffic['station'].unique()

    cnt_all = len(traffic)
    values = [((traffic['station'] == i).sum() *
               sum_budget) / float(cnt_all) for i in station_ids]
    values = iteround.saferound(values, places=0)

    return DictBudgetDist({i: int(val) for i, val in zip(station_ids, values)})
