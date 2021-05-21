import functools
import inspect
import itertools
import os
from typing import Callable, List, Optional, Union

import numpy as np
import pandas as pd

import lib.budget as bgt
import lib.simulation as sim
import lib.traffic as trf
import lib.utils as utils

from .. import simulation as sim
from . import utils as optutils
from .brute_force import optimize_brute_force
from .hard_assign import optimize_hard_assign
from .soft_assign import ModelSpec as SoftAssignModelSpec
from .soft_assign import build_model as build_soft_assign_model
from .soft_assign_heuristic import (optimize_soft_assign_heuristic,
                                    optimize_soft_assign_heuristic_single)

FILE_EXT = "pgz"


def _get_filename(dir: str, dataset_name: str, stations: List[int], is_result: bool):
    stations.sort()
    fname_parts = []
    if is_result:
        fname_parts.append('RESULT')
    fname_parts.append(dataset_name)
    fname_parts.extend(str(st) for st in stations)

    fname = '-'.join(fname_parts) + '.' + FILE_EXT
    return os.path.join(dir, fname)


class IncompleteSolutionException(Exception):
    def __init__(self, *args):
        super().__init__(*args)


def _check_complete_solution(
        log: pd.DataFrame,
        max_station_distance: float,
        solutions: optutils.Solutions,
        check_objective=True,
        allow_incomplete=False):
    max_budget = max(s.budget for s in utils.iter_values(solutions))
    max_objective = max(s.objective for s in utils.iter_values(solutions))

    if check_objective and max_objective != log.index.nunique():
        raise IncompleteSolutionException(
            f"best objective {max_objective} != {log.index.nunique()} (stations: {', '.join(str(st) for st in sorted(log.station.unique()))})")

    not_present = np.ones((max_budget + 1), dtype=int)
    not_present[0] = 0

    for k, sol in utils.iter_items(solutions, lambda sol: sol.budget):
        not_present[k] = 0

    if np.sum(not_present) > 0 and not allow_incomplete:
        idx_print = utils.int_list_printable(
            np.where(not_present > 0)[0].tolist())
        sts_print = utils.int_list_printable(sorted(log.station.unique()))

        raise IncompleteSolutionException(
            f"(stations: {sts_print}): missing {np.sum(not_present)}/{max_budget} solutions: {idx_print}")


def load_solution(
        spec: trf.TrafficSpec,
        log: pd.DataFrame,
        in_dir: str,
        name: str,
        check_objective: bool = True,
        partition: bool = True,
        max_n_stations_brute_force: int = 0):
    dists = []
    try:
        if partition:
            partitions = list(optutils.split_log_to_station_components(
                log, with_stations=True))
        else:
            partitions = [([], log)]

        for sts, sts_log in partitions:
            if 0 < len(sts) <= max_n_stations_brute_force:
                dists_this = optimize_brute_force(
                    sts_log, max_station_distance=spec.max_station_distance)
            else:
                fname = _get_filename(in_dir, name, sts, is_result=True)
                dists_this = utils.gzip_pickle_load(fname)
                if '-ilp-subset' in fname:
                    print(fname)
                if not optutils.is_wrapped(dists_this):
                    sts_plog = sim.PreprocessedLog.from_dataframe(sts_log)
                    dists_this = optutils.wrap_distributions(
                        dists_this)(
                            sts_plog, max_station_distance=spec.max_station_distance)
                _check_complete_solution(
                    sts_log, spec.max_station_distance, dists_this,
                    check_objective=check_objective, allow_incomplete=len(partitions) == 1)
            dists.append(dists_this)

        return optutils.merge_component_dists(dists)
    except (IncompleteSolutionException, FileNotFoundError) as e:
        print(name, "failed:", getattr(e, "message", repr(e)))
        return functools.partial(load_solution, spec,
                                 log, in_dir=in_dir, name=name,
                                 check_objective=check_objective,
                                 max_n_stations_brute_force=max_n_stations_brute_force)


def dump_default_soft_assign_ILP_opt_solution_request(
        spec: trf.TrafficSpec,
        log: pd.DataFrame,
        out_dir: str,
        in_dir: str,
        first_come_first_served: bool = False,
        max_n_stations_brute_force: int = 2):
    name = spec.to_canonical_id() + '-gurobi-model' + \
        ('-fcfs' if first_come_first_served else '')

    load_func = functools.partial(load_solution, spec,
                                  log, in_dir=in_dir, name=name, check_objective=first_come_first_served,
                                  max_n_stations_brute_force=max_n_stations_brute_force)

    result = load_func()

    if callable(result):
        max_station = log['station'].max()
        for sts, sts_log in optutils.split_log_to_station_components(log, with_stations=True):
            if len(sts) <= max_n_stations_brute_force:
                continue

            starter = optimize_soft_assign_heuristic_single(
                sts_log, spec.max_station_distance)[-1].distribution
            mspec = SoftAssignModelSpec(
                log=sts_log, budget_upper_limit=sum(starter),
                max_station_distance=spec.max_station_distance,
                opt_budget_for_full_coverage=True,
                starter_solution_distribution=bgt.ListBudgetDist(
                    starter).to_list(max_station),
                first_come_first_served=first_come_first_served
            )
            utils.gzip_pickle_dump(mspec, _get_filename(
                out_dir, name, sts, is_result=False))
        return load_func
    else:
        return result


def dump_heuristic_solution_request(
        spec: trf.TrafficSpec,
        log: pd.DataFrame,
        out_dir: str,
        in_dir: str,
        with_ilp_station_subset: bool = False,
        verbose: bool = False):
    name = spec.to_canonical_id() + '-heuristic'

    if with_ilp_station_subset:
        name += '-ilp-subset'

    load_func = functools.partial(load_solution, spec,
                                  log, in_dir=in_dir, name=name,
                                  partition=not with_ilp_station_subset)

    result = load_func()

    if callable(result):
        if with_ilp_station_subset:
            func = functools.partial(optimize_with_ilp_station_subset)

        if with_ilp_station_subset:
            func = functools.partial(optimize_with_ilp_station_subset,
                                     optimize_soft_assign_heuristic,
                                     spec=spec, verbose=verbose)
            utils.gzip_pickle_dump(func, _get_filename(
                out_dir, name, [], is_result=False))
        else:
            for sts, sts_log in optutils.split_log_to_station_components(log, with_stations=True):
                func = functools.partial(optimize_soft_assign_heuristic, sts_log,
                                         max_station_distance=spec.max_station_distance)

                utils.gzip_pickle_dump(func, _get_filename(
                    out_dir, name, sts, is_result=False))

        return load_func
    else:
        return result


def _find_optimal_station_subset(
        spec: trf.TrafficSpec,
        build_traffic: Callable[[trf.TrafficSpec], pd.DataFrame],
        verbose=False):
    if verbose:
        print("Building ILP station reduction traffic...")

    subset_spec = trf.TrafficSpec(start_timestamp=spec.start_timestamp,
                                  end_timestamp=spec.end_timestamp,
                                  max_station_distance=spec.max_station_distance,
                                  max_stations_pair_distance=spec.max_stations_pair_distance,
                                  max_attempts=None,
                                  station_subset=spec.station_subset,
                                  minimal_complete_station_subset=False,
                                  subsequent_attempts_by_original_position=True,
                                  remove_redundant_station_attempts=False)
    traffic = build_traffic(subset_spec)
    log = sim.traffic_to_log(traffic)

    mspec = SoftAssignModelSpec(log=log,
                                max_station_distance=spec.max_station_distance,
                                budget_upper_limit=1,
                                opt_budget_for_full_coverage=True,
                                secondary_opt_distance_sum=True,
                                station_subset_only_mode=True)

    dist = None
    if verbose:
        print("Optimizing ILP station reduction model...")
    with build_soft_assign_model(mspec) as mb:
        dist = mb.optimize(
            compute_iis=True, return_as_solution=False, return_simulation=False)
    sts = [i for i, v in enumerate(dist) if v >= 1]
    return sts


def optimize_with_ilp_station_subset(
        optimize_func: Callable[..., optutils.Solutions],
        spec: trf.TrafficSpec,
        build_traffic: Callable[[trf.TrafficSpec], pd.DataFrame],
        verbose=False,
        **explicit_kwargs):

    sts = _find_optimal_station_subset(spec, build_traffic, verbose=verbose)

    func_param_names = inspect.signature(optimize_func).parameters

    spec = spec.copy(station_subset=sts)

    kwargs = dict(
        spec=spec,
        build_traffic=build_traffic,
        max_station_distance=spec.max_station_distance,
        verbose=verbose
    )

    kwargs = {k: v for k, v in kwargs.items() if k in func_param_names}
    kwargs.update(explicit_kwargs)

    if 'log' in func_param_names:
        if verbose:
            print("Building station subset traffic...")

        traffic = build_traffic(spec)
        log = sim.traffic_to_log(traffic)
        kwargs['log'] = log

    return optimize_func(**kwargs)
