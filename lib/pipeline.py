import dataclasses
import functools
import inspect
import itertools
from dataclasses import dataclass
from multiprocessing import Pool, cpu_count
from typing import (Any, Callable, Dict, Iterable, List, NamedTuple, Optional,
                    Set, Tuple, TypeVar, Union)

import pandas as pd

import lib.optimize as opt
import lib.simulation as sim
import lib.traffic as trf
import lib.utils as utils


def _return_true(*kargs, **kwargs):
    return True


def build_all_traffics(
        specs: List[trf.TrafficSpec],
        threads: Optional[int] = None, progress=False,
        **build_traffic_kwargs) -> Dict[trf.TrafficSpec, pd.DataFrame]:
    threads = cpu_count() if threads is None else threads
    p = Pool(threads) if threads > 1 else None

    result = dict()

    try:
        func = functools.partial(trf.build_traffic, **build_traffic_kwargs)
        func = functools.partial(utils.map_value, func)
        spec_items = [(spec, spec) for spec in specs]
        it = p.imap(func, spec_items) if p is not None else map(
            func, spec_items)
        result = {spec: tr for spec, tr in utils.progressify(
            it, "Specs", enabled=progress, total=len(specs))}
    finally:
        if p is not None:
            p.close()

    return result


@dataclass(frozen=True, repr=False)
class Solver:
    name: str
    callable: Callable[..., Any] = dataclasses.field(compare=False)
    traffic_filter: Callable[[trf.TrafficSpec, pd.DataFrame], bool] = dataclasses.field(
        default=_return_true, compare=False)
    wrap_solution_after: bool = False
    is_multi_cpu: bool = False

    def __repr__(self):
        return f"Solver('{self.name}')"

    def __getstate__(self):
        return (self.name, self.callable, self.wrap_solution_after, self.is_multi_cpu)

    def __setstate__(self, state):
        name, c, wrap_solution_after, is_multi_cpu = state
        object.__setattr__(self, 'name', name)
        object.__setattr__(self, 'callable', c)
        object.__setattr__(self, 'wrap_solution_after', wrap_solution_after)
        object.__setattr__(self, 'is_multi_cpu', is_multi_cpu)


class Solvable(NamedTuple):
    traffic_spec: trf.TrafficSpec
    solver: Solver

    def __repr__(self):
        return f"Solvable({self.traffic_spec}, {self.solver})"


def assign_solvers(traffics: Dict[trf.TrafficSpec, pd.DataFrame], solvers: Iterable[Solver]) -> List[Solvable]:
    result = []

    for solver in solvers:
        for spec, tr in traffics.items():
            if solver.traffic_filter(spec, tr):
                result.append(Solvable(spec, solver))

    return result


def solve_single(spec: trf.TrafficSpec, traffic: pd.DataFrame, build_traffic: Callable[[trf.TrafficSpec], pd.DataFrame], solver: Solver, **kwargs):
    insp = inspect.signature(solver.callable)

    call_args = dict()

    log = None
    if 'log' in insp.parameters:
        log = sim.traffic_to_log(traffic)
        call_args['log'] = log

    if 'spec' in insp.parameters:
        call_args['spec'] = spec
    elif 'max_station_distance' in insp.parameters:
        call_args['max_station_distance'] = spec.max_station_distance

    if 'build_traffic' in insp.parameters:
        call_args['build_traffic'] = build_traffic

    for k, v in kwargs.items():
        if k in insp.parameters:
            call_args[k] = v

    result = solver.callable(**call_args)

    if solver.wrap_solution_after:
        if log is None:
            log = sim.traffic_to_log(traffic)
        plog = sim.PreprocessedLog.from_dataframe(log)
        result = opt.utils.wrap_distributions(result)(
            plog, max_station_distance=spec.max_station_distance)

    return result


def _resolve_callable(solvable_callable_tuple):
    solvable, c = solvable_callable_tuple
    return solvable, c()


def _resolve_single(solvable: Solvable, traffics: Dict[trf.TrafficSpec, pd.DataFrame], build_traffic: Callable[[trf.TrafficSpec], pd.DataFrame], **kwargs):
    if solvable.traffic_spec in traffics:
        traffic = traffics[solvable.traffic_spec]
    else:
        traffic = build_traffic(solvable.traffic_spec)

    return solvable, solve_single(solvable.traffic_spec, traffic, build_traffic, solvable.solver, **kwargs)


def update_solutions(
        solutions: Dict[Solvable, Any],
        solvables: List[Solvable],
        traffics: Dict[trf.TrafficSpec, pd.DataFrame],
        build_traffic: Callable[[trf.TrafficSpec], pd.DataFrame],
        threads: Optional[int] = None,
        progress=False,
        multi_cpu_kwargs: Dict[str, Any] = dict(),
        single_cpu_kwargs: Dict[str, Any] = dict(),
        **kwargs):
    threads = cpu_count() if threads is None else threads
    p = Pool(threads) if threads > 1 else None
    nunresolved = 0

    try:
        mapfunc = p.imap_unordered if p is not None else map

        callables = {s for s, r in solutions.items() if callable(r)}
        solved = {s for s, r in solutions.items() if not not r}

        serials = {s for s in solvables if not s.solver.is_multi_cpu}
        parallels = {s for s in solvables if s.solver.is_multi_cpu}

        serials.difference_update(solved)
        parallels.difference_update(solved)

        callables.intersection_update(solvables)
        callables = [(s, solutions[s]) for s in callables]
        if len(callables) >= 1:
            it = mapfunc(_resolve_callable, callables)
            for solvable, result in utils.progressify(
                    it, "Callable", enabled=progress, total=len(callables)):
                solutions[solvable] = result
                if callable(result):
                    nunresolved += 1

        func = functools.partial(
            _resolve_single, traffics=traffics, build_traffic=build_traffic, **single_cpu_kwargs, **kwargs)
        it = mapfunc(func, serials)
        for solvable, result in utils.progressify(
                it, "Serial", enabled=progress, total=len(serials)):
            solutions[solvable] = result
            if callable(result):
                nunresolved += 1

        for solvable in utils.progressify(
                parallels, "Parallel", enabled=progress, total=len(parallels)):
            solutions[solvable] = solve_single(
                solvable.traffic_spec,
                traffics[solvable.traffic_spec],
                solvable.solver, **multi_cpu_kwargs, **kwargs)
    finally:
        print("Unresolved solutions:", nunresolved)
        if p is not None:
            p.close()


_T = TypeVar('_T')


class CrossvalidationRequestKey(NamedTuple):
    val_traffic_spec: trf.TrafficSpec
    solvable: Solvable
    simulation_spec: trf.SimulationSpec
    is_auto_attempt_simulation: bool

    def __repr__(self):
        return f"CrossvalReqKey({self.val_traffic_spec}, {self.solvable}, {self.simulation_spec}, auto_attempt_simulation={self.is_auto_attempt_simulation})"


def crossvalidate_all(
        requests: Dict[CrossvalidationRequestKey, opt.utils.Distributions],
        post_func: Callable[[pd.DataFrame], _T],
        traffic_or_log_provider: Callable[[
            trf.TrafficSpec], Union[pd.DataFrame, sim.PreprocessedLog]],
        auto_attempts: bool = True,
        progress=False,
        **kwargs) -> Dict[Tuple, _T]:
    func = opt.utils.simulate_with_auto_attempts_all if auto_attempts else opt.utils.simulate_all

    results = dict()

    for key, dists in utils.progressify(requests.items(), "Instances", enabled=progress):
        result_this = func(traffic_or_log_provider(
            key.val_traffic_spec), dists, post_func=post_func, **key.simulation_spec.to_dict(), **kwargs)
        results[key] = result_this

    return results


def crossvalidate_all_multithread(
        requests: Dict[CrossvalidationRequestKey, opt.utils.Distributions],
        post_func: Callable[[pd.DataFrame], _T],
        traffic_or_log_provider: Callable[[
            trf.TrafficSpec], Union[pd.DataFrame, sim.PreprocessedLog]],
        auto_attempts: bool = True,
        progress=False, threads: Optional[int] = None,
        **kwargs) -> Dict[Tuple, _T]:
    threads = cpu_count() if threads is None else threads

    assert threads >= 1

    if threads == 1:
        return crossvalidate_all(requests, post_func, traffic_or_log_provider, auto_attempts=auto_attempts, progress=progress, **kwargs)

    requests = utils.chunks(requests, threads)

    with Pool(threads) as p:
        results_all = dict()

        func = functools.partial(
            crossvalidate_all, post_func=post_func, traffic_or_log_provider=traffic_or_log_provider,
            auto_attempts=auto_attempts, progress=progress, **kwargs)

        for result in utils.progressify(p.imap_unordered(func, requests), "Threads", enabled=progress, total=threads):
            results_all.update(result)

        return results_all
