import functools
import itertools
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from .. import simulation as sim
from .. import utils
from ..budget import BudgetDist, DictBudgetDist, ListBudgetDist
from . import utils as optutils


def find_ideal_budget_dist(log: pd.DataFrame, max_station: int) -> ListBudgetDist:
    """Find smallest budget for a traffic log dataframe that fully satisfies the demand
    (using the closest station for each charging request).
    For each station, this is the maximum amount of cars present at the station
    at once over the entire simulation (assuming charging at the closest station).

    Only makes sense for logs without additional attempts.

    :param log: The traffic log dataframe, as created by `sim.traffic_to_log()`
    :type  log: pd.DataFrame
    :param max_station: Maximum station value
    :type  max_station: int
    :rtype: ListBudgetDist
    """
    return ListBudgetDist(log.groupby('station')['status']
                          .apply(lambda x: x.cumsum().max())
                          .reindex(pd.RangeIndex(max_station + 1), fill_value=0)
                          .tolist())


def optimize_hard_assign_legacy(
        log: pd.DataFrame,
        max_station_distance: float,
        progress=False):
    solutions_all: List[optutils.Solutions] = []
    max_idx = log.index.max()
    for station in utils.progressify(log['station'].unique(), "Stations", enabled=progress):
        st_log = log[log['station'] == station].copy()
        st_plog = sim.PreprocessedLog.from_dataframe(st_log)
        max_obj = st_log.index.nunique()

        solutions_this = []
        for budget in itertools.count(1):
            dist = DictBudgetDist({station: budget})
            objective = sim.get_satisfied_charging_requests(
                sim.simulate(
                    st_plog, dist,
                    max_station_distance=max_station_distance,
                    max_station=station, max_idx=max_idx))

            solutions_this.append(optutils.Solution(
                budget=budget,
                distribution=dist,
                objective=objective
            ))

            if objective == max_obj:
                break

        solutions_all.append(solutions_this)

    return optutils.merge_component_dists_bad(solutions_all)


def optimize_hard_assign(
        log: pd.DataFrame,
        max_station_distance: float,
        budget_max: Optional[int] = None,
        progress=False):
    solutions_all: List[optutils.Solutions] = []
    max_idx = log.index.max()
    for station in utils.progressify(log['station'].unique(), "Stations", enabled=progress):
        st_log = log[log['station'] == station].copy()
        st_plog = sim.PreprocessedLog.from_dataframe(st_log)
        max_obj = st_log.index.nunique()

        solutions_this = []
        it = itertools.count(1) if budget_max is None else range(1, budget_max + 1)
        for budget in it:
            dist = DictBudgetDist({station: budget})
            objective = sim.get_satisfied_charging_requests(
                sim.simulate(
                    st_plog, dist,
                    max_station_distance=max_station_distance,
                    max_station=station, max_idx=max_idx))

            solutions_this.append(optutils.Solution(
                budget=budget,
                distribution=dist,
                objective=objective
            ))

            if objective == max_obj:
                break

        solutions_all.append(solutions_this)

    return optutils.merge_component_dists(solutions_all, budget_max=budget_max)
