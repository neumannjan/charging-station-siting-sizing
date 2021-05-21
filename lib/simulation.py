import copy
from dataclasses import dataclass
from datetime import timedelta
from typing import Callable, Dict, List, NamedTuple, Optional

import iteround
import numpy as np
import pandas as pd

import fleet_electrification_cpp as cpp

from .budget import BudgetDist, DictBudgetDist, ListBudgetDist, BudgetDistLike
from . import clustering
from . import traffic as trf

def sort_log(log: pd.DataFrame, traffic: Optional[pd.DataFrame] = None, inplace=False):
    if not inplace:
        log = log.copy()

    additional_cols = ['attempt_no']
    additional_cols = [c for c in additional_cols if c in log.columns]
    log.sort_values(['timestamp', 'status', *additional_cols], inplace=True)
    log['ord'] = np.arange(len(log))
    log['opposite_ord'] = 0

    st_mask = log['status'] == 1
    if 'attempt_no' in log:
        for att in log['attempt_no'].unique():
            att_mask = log['attempt_no'] == att
            log.loc[att_mask & st_mask, 'opposite_ord'] = \
                log.loc[att_mask & (~st_mask), 'ord']

            log.loc[att_mask & (~st_mask), 'opposite_ord'] = \
                log.loc[att_mask & st_mask, 'ord']
    else:
        log.loc[st_mask, 'opposite_ord'] = log.loc[~st_mask, 'ord']
        log.loc[~st_mask, 'opposite_ord'] = log.loc[st_mask, 'ord']

    if not inplace:
        return log


def join_log_traffic(log: pd.DataFrame, traffic: Optional[pd.DataFrame]):
    if traffic is not None:
        return log.merge(traffic, how='left', left_index=True, right_index=True, suffixes=('', '_original'))
    return log.copy()


def traffic_to_log(traffic: pd.DataFrame):
    opt_cols = ['station_distance', 'attempt_no', 'penalty']
    arrs = traffic[['arrival', 'station', 'vehicle', *
                    [c for c in opt_cols if c in traffic.columns]]].copy()
    arrs.rename(columns={'arrival': 'timestamp'}, inplace=True)
    arrs['status'] = 1
    deps = traffic[['departure', 'station', 'vehicle', *
                    [c for c in opt_cols if c in traffic.columns]]].copy()
    deps.rename(columns={'departure': 'timestamp'}, inplace=True)
    deps['status'] = -1

    log = pd.concat([arrs, deps])
    del arrs
    del deps

    sort_log(log, inplace=True)

    return log


def distances_to_likelihood_quad(distances: List[float]):
    return [1 / (d * d) for d in distances]


def distance_to_time_none(distance: float):
    return 0


class PreprocessedLog:
    log: np.ndarray
    dtype = [
        ('idx', np.int64),
        ('timestamp', np.int64),
        ('vehicle', np.int32),
        ('station', np.int16),
        ('station_distance', np.float64),
        ('penalty', np.float64),
        ('attempt_no', np.int8),
        ('status', np.int8),
    ]
    datetime_cols = ['timestamp']

    def __init__(self, log: np.ndarray):
        self.log = log

    def to_dataframe(self, traffic: Optional[pd.DataFrame] = None):
        df = pd.DataFrame(self.log)
        df.set_index('idx', inplace=True)

        for c in PreprocessedLog.datetime_cols:
            df[c] = pd.to_datetime(df[c] * 1e9)

        if traffic is not None:
            df = join_log_traffic(df, traffic)

        return df

    @staticmethod
    def from_dataframe(log: pd.DataFrame):
        log.index.name = 'idx'

        if 'penalty' not in log.columns:
            log['penalty'] = 0
        if 'attempt_no' not in log.columns:
            log['attempt_no'] = 0

        plog = PreprocessedLog(
            np.array(log[[v[0] for v in PreprocessedLog.dtype[1:]]].to_records(),
                     dtype=PreprocessedLog.dtype))

        for c in PreprocessedLog.datetime_cols:
            plog.log[c] = plog.log[c] // 1e9

        return plog

    def apply(self, func):
        return PreprocessedLog(func(self.log))

    def __getitem__(self, mask):
        return PreprocessedLog(self.log[mask])

    def __repr__(self):
        return repr(self.to_dataframe())


class PreprocessedTraffic:
    traffic: np.ndarray
    dtype = [
        ('idx', np.int64),
        ('arrival', np.int64),
        ('departure', np.int64),
        ('vehicle', np.int32),
        ('station', np.int16),
        ('station_distance', np.float64),
        ('penalty', np.float64),
        ('attempt_no', np.int8),
    ]
    datetime_cols = ['arrival', 'departure']

    def __init__(self, traffic: np.ndarray):
        self.traffic = traffic

    def to_dataframe(self):
        df = pd.DataFrame(self.traffic)
        df.set_index('idx', inplace=True)

        for c in PreprocessedTraffic.datetime_cols:
            df[c] = pd.to_datetime(df[c] * 1e9)

        return df

    @staticmethod
    def from_dataframe(traffic: pd.DataFrame):
        traffic.index.name = 'idx'

        if 'penalty' not in traffic.columns:
            traffic['penalty'] = 0
        if 'attempt_no' not in traffic.columns:
            traffic['attempt_no'] = 0

        ptraffic = PreprocessedTraffic(
            np.array(traffic[[v[0] for v in PreprocessedTraffic.dtype[1:]]].to_records(),
                     dtype=PreprocessedTraffic.dtype))

        for c in PreprocessedTraffic.datetime_cols:
            ptraffic.traffic[c] = ptraffic.traffic[c] // 1e9

        return ptraffic

    def __repr__(self):
        return repr(self.to_dataframe())


def add_attempts_to_traffic(ptraffic: PreprocessedTraffic,
                            station_distances_mtx: List[List[float]],
                            max_attempts: Optional[int], max_stations_pair_distance: float,
                            station_choice_probabilistic: bool,
                            distance_to_time: Callable[[float], int] = distance_to_time_none,
                            distances_to_likelihood: Callable[[List[float]], List[float]] = distances_to_likelihood_quad):
    if max_attempts is None:
        max_attempts = len(np.unique(ptraffic.traffic['station']))

    traffic_new = cpp.add_attempts_to_traffic(ptraffic.traffic, station_distances_mtx, max_attempts,
                                              max_stations_pair_distance, station_choice_probabilistic,
                                              distance_to_time, distances_to_likelihood)
    return PreprocessedTraffic(traffic_new).to_dataframe()


def simulate(plog: PreprocessedLog, budget_distribution: BudgetDistLike,
             max_station_distance: float, max_station: Optional[int] = None,
             max_idx: Optional[int] = None, reindex=True):
    """Simulates traffic.

    Returns new, updated 'station' column (`pd.Series`) of a traffic dataframe,
    where each entry contains the station at which the vehicle is charged
    (at one of its attempts), or a negative value if rejected completely.

    The returned column can be added to the original traffic dataframe
    (the dataframe without added attempts).

    :param plog: The traffic log dataframe, in preprocessed form.
    :type  plog: PreprocessedLog
    :param budget_distribution: Charger distribution for each charging station
    :type  budget_distribution: BudgetDistLike
    :param max_station_distance: Maximum distance between vehicle and closest station
    :type  max_station_distance: float
    :param max_station: Maximum station value
    :type  max_station: Optional[int]
    :param max_idx: Maximum idx value
    :type  max_idx: Optional[int]
    """

    if not isinstance(budget_distribution, BudgetDist):
        budget_distribution = ListBudgetDist(budget_distribution)

    if max_station is None:
        max_station = plog.log['station'].max()

    budget_distribution = budget_distribution.to_list(max_station)

    if max_idx is not None:
        max_idx = int(max_idx)

    satisfied = cpp.simulate(
        plog.log, budget_distribution, max_station_distance, max_idx)

    if reindex:
        idx = pd.Index(plog.log['idx'], name='idx').drop_duplicates()
        return pd.Series(satisfied).reindex(idx)
    else:
        return satisfied


def independent_station_components(log: PreprocessedLog):
    return cpp.independent_station_components(log.log)


def get_simulation_coverage(station_satisf_series: pd.Series) -> float:
    """Get coverage ratio for simulation result

    :param station_satisf_series:  Pandas series returned by `simulate()`
    :type  station_satisf_series:  pd.Series

    :rtype :  float
    """

    return (station_satisf_series >= 0).mean()


def get_satisfied_charging_requests(station_satisf_series: pd.Series) -> int:
    """Get amount of satisfied charging requests for simulation result

    :param station_satisf_series:  Pandas series returned by `simulate()`
    :type  station_satisf_series:  pd.Series

    :rtype :  int
    """

    return (station_satisf_series >= 0).sum()


def simulate_with_auto_attempts(
        traffic: pd.DataFrame, budget_distribution: BudgetDistLike,
        max_station_distance: float, max_stations_pair_distance: float, max_attempts: Optional[int],
        subsequent_attempts_by_original_position: bool,
        charging_stations: pd.DataFrame, station_distances_mtx: List[List[float]]):
    """
    Simulate such that the original charging attempt,
    as well as additional follow-up attempts
    are decided upon solely based on the charger distribution
    -> based on which stations are actually built.

    Calls `simulate` after the attempts are found.
    """

    # choose from stations that actually are built -> have at least 1 charger
    charging_stations = charging_stations.loc[[
        i for i, v in enumerate(budget_distribution) if v >= 1]]

    traffic_sim = trf.build_traffic(
        trf.TrafficSpec(
            max_station_distance=max_station_distance,
            max_stations_pair_distance=max_stations_pair_distance,
            max_attempts=max_attempts,
            station_subset=list(charging_stations.index),
            minimal_complete_station_subset=False,
            subsequent_attempts_by_original_position=subsequent_attempts_by_original_position,
            remove_redundant_station_attempts=False
        ),
        traffic_full=traffic,
        charging_stations=charging_stations,
        station_distances_mtx=station_distances_mtx)

    log = traffic_to_log(traffic_sim)
    plog = PreprocessedLog.from_dataframe(log)
    return simulate(plog, budget_distribution, max_station_distance=max_station_distance)
