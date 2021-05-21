from typing import Iterable, List, Optional
import itertools

import numpy as np
import pandas as pd

import lib.simulation as sim
import lib.budget as bgt
from . import utils as optutils


def all_distributions_of(stations: List[int], total_budget: int) -> Iterable[bgt.ListBudgetDist]:
    if total_budget == 0:
        yield bgt.ListBudgetDist([])
    elif len(stations) > 0:
        yield from all_distributions_of(stations[1:], total_budget)

        for i in range(1, total_budget + 1):
            for d in all_distributions_of(stations[1:], total_budget - i):
                yield bgt.DictBudgetDist({stations[0]: i}) + d


@optutils.station_component_wise
def optimize_brute_force(log: pd.DataFrame, max_station_distance: float):
    stations = log['station'].unique()
    stations.sort()

    idx_total = log.index.nunique()
    max_station = max(stations)
    max_idx = log.index.max()

    results = []

    plog = sim.PreprocessedLog.from_dataframe(log)

    for budget in itertools.count(1):
        result_best = 0
        dist_best: bgt.ListBudgetDist = bgt.ListBudgetDist([])

        for d in all_distributions_of(stations, budget):
            result_this = sim.get_satisfied_charging_requests(
                sim.simulate(
                    plog, d,
                    max_station_distance=max_station_distance,
                    max_station=max_station, max_idx=max_idx))

            if result_this > result_best:
                result_best = result_this
                dist_best = d

                if result_this == idx_total:
                    break

        results.append(optutils.Solution(
            budget=budget,
            distribution=dist_best,
            objective=result_best))

        if result_this == idx_total:
            break

    return results
