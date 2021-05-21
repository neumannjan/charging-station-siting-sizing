import functools
import inspect
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Union, TypeVar, Optional

import fleet_electrification_cpp as cpp
import numpy as np
import pandas as pd

import lib.budget as bgt
import lib.simulation as sim
import lib.utils as utils

_T = TypeVar('_T')

Distributions = Union[Dict[int, bgt.BudgetDistLike],
                      List[bgt.BudgetDistLike]]


@dataclass
class Solution:
    budget: int
    distribution: bgt.BudgetDistLike
    objective: int

    def to_dict(self):
        return dict(self.__dict__)


Solutions = Union[Dict[int, Solution], List[Solution]]


def wrap_distributions(distributions: Distributions):
    def wrapper(plog: sim.PreprocessedLog, max_station_distance: float, **sim_kwargs) -> Dict[int, Solution]:
        dists_new = dict()
        for b, d in utils.iter_items(distributions, sum):
            dist = bgt.ListBudgetDist(d)

            dists_new[b] = Solution(
                b, dist,
                sim.get_satisfied_charging_requests(sim.simulate(
                    plog, dist, max_station_distance, **sim_kwargs)))

        return dists_new

    return wrapper


def unwrap_distributions(solutions: Solutions, copy=True) -> Dict[int, bgt.ListBudgetDist]:
    if copy:
        return {s.budget: bgt.ListBudgetDist(s.distribution) for s in utils.iter_values(solutions)}
    else:
        return {s.budget: s.distribution for s in utils.iter_values(solutions)}


def is_wrapped(solutions: Union[Solutions, Distributions]) -> bool:
    return isinstance(next(iter(utils.iter_values(solutions)), None), Solution)


def merge_component_dists_bad(solutions_all: List[Solutions]) -> List[Solution]:
    if len(solutions_all) == 1:
        return solutions_all[0]

    result = []
    solutions_all = [dict(utils.iter_items(ss, lambda s: s.budget))
                     for ss in solutions_all]
    used = [max(ss.keys()) for ss in solutions_all]
    dist = functools.reduce(
        lambda x, y: x+y, [ss[u].distribution if u > 0 else bgt.ListBudgetDist([]) for ss, u in zip(solutions_all, used)])

    result.append(Solution(sum(dist), dist, sum(
        ss[u].objective if u > 0 else 0 for ss, u in zip(solutions_all, used))))

    for budget in utils.progressify(range(sum(dist)-1, 0, -1), "Merge", enabled=True):
        new_dists: List[bgt.BudgetDistLike] = [
            bgt.ListBudgetDist([]) for _ in range(len(used))]
        results = [0 for _ in range(len(used))]
        for i in range(len(used)):
            if used[i] == 0:
                continue

            used[i] -= 1
            new_dists[i] = functools.reduce(
                lambda x, y: x+y, [ss[u].distribution if u > 0 else bgt.ListBudgetDist([]) for ss, u in zip(solutions_all, used)])
            results[i] = sum(ss[u].objective if u > 0 else 0 for ss,
                             u in zip(solutions_all, used))
            used[i] += 1

        i_best = np.argmax(results)
        used[i_best] -= 1
        result.append(Solution(budget, new_dists[i_best], results[i_best]))

    result.reverse()
    return result


def merge_component_dists(solutions_all: List[Solutions], budget_max: Optional[int] = None) -> List[Solution]:
    if len(solutions_all) == 1:
        return solutions_all[0]

    solutions_all_d: List[Dict[int, Solution]] = [
        {k: v for k, v in utils.iter_items(s, lambda v: v.budget)} for s in solutions_all]

    @functools.lru_cache(maxsize=None)
    def func(i: int, budget: int):
        if i >= len(solutions_all_d) or budget == 0:
            return 0, bgt.DictBudgetDist(dict())
        if budget < 0:
            raise Exception()

        max_obj, max_dist = func(i+1, budget)

        for budget_this in range(1, budget+1):
            obj, dist = func(i+1, budget - budget_this)
            if budget_this in solutions_all_d[i]:
                obj += solutions_all_d[i][budget_this].objective
            else:
                continue

            if obj > max_obj:
                max_obj = obj
                max_dist = dist + solutions_all_d[i][budget_this].distribution

        return max_obj, max_dist

    results = []

    budget_max_found = sum(max(solutions.keys())
                           for solutions in solutions_all_d)
    if budget_max is not None:
        budget_max = min(budget_max_found, budget_max)
    else:
        budget_max = budget_max_found

    for budget in utils.progressify(range(1, budget_max + 1), "Merge", enabled=True):
        obj, dist = func(0, budget)
        results.append(
            Solution(budget=budget, distribution=dist, objective=obj))

    return results


def split_log_to_station_components(log: pd.DataFrame, copy=True, progress=False, with_stations=False):
    plog = sim.PreprocessedLog.from_dataframe(log)

    components = cpp.independent_station_components(plog.log)
    for sts in utils.progressify(components, "Components", enabled=progress and len(components) > 1):
        log_this = log[log['station'].isin(sts)]
        if copy:
            log_this = log_this.copy()

        if with_stations:
            yield sts, log_this
        else:
            yield log_this


def station_component_wise(func):
    wants_progress = 'progress' in inspect.signature(func).parameters

    @functools.wraps(func)
    def wrapper(log: pd.DataFrame, max_station_distance: float, **kwargs):
        if wants_progress:
            progress = kwargs.get('progress', False)
        else:
            progress = kwargs.pop('progress', False)

        results = []
        for l in split_log_to_station_components(log, progress=progress):
            result = func(
                l, max_station_distance=max_station_distance, **kwargs)
            results.append(result)

        if len(results) == 0:
            return []
        elif len(results) == 1:
            return results[0]
        elif is_wrapped(results[0]):
            plog = sim.PreprocessedLog.from_dataframe(log)
            return merge_component_dists(results)
        else:
            plog = sim.PreprocessedLog.from_dataframe(log)
            return unwrap_distributions(merge_component_dists(
                [wrap_distributions(r)(plog, max_station_distance)
                 for r in results]
            ), copy=False)

    return wrapper


def _nomod(df: pd.DataFrame) -> pd.DataFrame:
    return df


def simulate_all(plog: sim.PreprocessedLog, distributions: Distributions, post_func: Callable[[pd.DataFrame], _T] = _nomod, progress=False, **kwargs) -> Dict[int, _T]:
    result = dict()

    for k, d in utils.progressify(utils.iter_items(distributions, sum), "Simulations", total=len(distributions), enabled=progress):
        result[k] = post_func(sim.simulate(plog, d, **kwargs))

    return result


def simulate_with_auto_attempts_all(traffic: pd.DataFrame, distributions: Distributions, post_func: Callable[[pd.DataFrame], _T] = _nomod, progress=False, **kwargs) -> Dict[int, _T]:
    result = dict()

    for k, d in utils.progressify(utils.iter_items(distributions, sum), "Simulations", total=len(distributions), enabled=progress):
        result[k] = post_func(sim.simulate_with_auto_attempts(
            traffic, d, **kwargs))

    return result
