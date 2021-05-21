import functools
import itertools
from multiprocessing import Pool, cpu_count
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

import lib.budget as bgt
import lib.simulation as sim
import lib.traffic as trf
import lib.utils as utils

from .utils import Solution, station_component_wise


def _optimize_soft_assign_heuristic_single(
        log: pd.DataFrame,
        max_station_distance: float,
        budget_max: Optional[int] = None,
        progress=False) -> List[Solution]:
    plog = sim.PreprocessedLog.from_dataframe(log)
    max_station = log['station'].max()
    max_idx = log.index.max()
    idx_arr = log.index.to_numpy()

    results: List[Solution] = []

    log_unresolved = log
    prev_distribution: Dict[int, int] = dict()

    it = (range(1, budget_max+1) if budget_max is not None else itertools.count(1))
    for budget in utils.progressify(it, "Iterations", enabled=progress):
        if budget_max is not None:
            mask = (log_unresolved.groupby('station')[
                    'status'].cumsum() <= (budget_max - budget + 1)).to_numpy()
            log_unresolved = log_unresolved[mask]

        stations = log_unresolved.query(
            'status == 1').reset_index().groupby('station')['idx'].nunique()
        if len(stations) == 0:
            break
        station = stations.idxmax()
        if station in prev_distribution:
            prev_distribution[station] += 1
        else:
            prev_distribution[station] = 1
        dist = bgt.DictBudgetDist(prev_distribution)

        sim_out = sim.simulate(
            plog, dist,
            max_station_distance=max_station_distance,
            max_station=max_station, max_idx=max_idx, reindex=False)
        objective = sim.get_satisfied_charging_requests(sim_out)

        results.append(Solution(budget, dist, objective))
        log_unresolved = log[(np.array(sim_out) <= -1)[idx_arr]]

    return results


@station_component_wise
def optimize_soft_assign_heuristic_single(
        log: pd.DataFrame,
        max_station_distance: float,
        budget_max: Optional[int] = None,
        progress=False):
    return _optimize_soft_assign_heuristic_single(
        log, max_station_distance, budget_max, progress)


@station_component_wise
def optimize_soft_assign_heuristic(
        log: pd.DataFrame,
        max_station_distance: float,
        start=1,
        end=float('inf'),
        threads: Optional[int] = None,
        progress=False) -> List[Solution]:
    threads: int = cpu_count() if threads is None else threads

    results: List[Solution] = []

    func = functools.partial(
        _optimize_soft_assign_heuristic_single, log, max_station_distance)

    n_full = log.index.nunique()

    done = False
    max_budget = float('inf')
    p = Pool(threads) if threads > 1 else None
    try:
        for budget_max in utils.progressify(
                itertools.count(start, step=threads), "Iterations", enabled=progress):
            if budget_max > end:
                break

            it: Iterable[List[Solution]]
            if p is not None:
                it = p.imap_unordered(func, range(
                    budget_max, budget_max+threads))
            else:
                it = [func(budget_max)]
            for results_this in it:
                for i, sol in enumerate(results_this):
                    if sol.budget > max_budget:
                        break

                    prev_objective = results[i].objective if len(
                        results) > i else 0
                    if sol.objective > prev_objective:
                        if len(results) > i:
                            results[i] = sol
                        else:
                            results.append(sol)

                        if sol.objective == n_full:
                            done = True
                            results = results[:i+1]
                            max_budget = i
                            continue

            if done:
                break
    except Exception as e:
        raise Exception() from e
    finally:
        if p is not None:
            p.close()

    return results
