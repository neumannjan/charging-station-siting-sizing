import warnings
from dataclasses import dataclass
import dataclasses
from datetime import datetime, date
from typing import List, Optional, Union

import numpy as np
import pandas as pd

import lib.clustering as clustering
import lib.simulation as sim


@dataclass(frozen=True)
class TrafficColumns:
    index: str = 'idx'
    arrival: str = 'arrival'
    departure: str = 'departure'
    station: str = 'station'
    vehicle: str = 'vehicle'
    attempt_no: str = 'attempt_no'
    station_distance: str = 'station_distance'
    x: str = 'x'
    y: str = 'y'
    lon: str = 'lon'
    lat: str = 'lat'


def _to_datetime(timestamp):
    if isinstance(timestamp, str):
        return datetime.fromisoformat(timestamp)
    if isinstance(timestamp, date):
        return datetime.combine(timestamp, datetime.min.time())


@dataclass(frozen=True)
class SimulationSpec:
    name: Optional[str] = dataclasses.field(compare=False, default=None)

    max_station_distance: float = float('inf')
    max_stations_pair_distance: float = float('inf')
    max_attempts: Optional[int] = 1

    subsequent_attempts_by_original_position: bool = False

    def copy(self, **kwargs):
        return dataclasses.replace(self, **kwargs)

    def to_canonical_id(self, named=False):
        out = [
            self.name if named else None,
            f"{self.max_attempts}att" if self.max_attempts is not None else None,
            str(self.max_station_distance),
            str(self.max_stations_pair_distance),
            'origpos' if self.subsequent_attempts_by_original_position else None,
        ]

        return '-'.join([o for o in out if o is not None])

    def __repr__(self):
        vals = [
            self.name,
            self.to_canonical_id(named=False)
        ]

        vals = ', '.join([repr(v) for v in vals if v is not None])
        return f"SimulationSpec({vals})"

    def to_dict(self):
        return {
            'max_station_distance': self.max_station_distance,
            'max_stations_pair_distance': self.max_stations_pair_distance,
            'max_attempts': self.max_attempts,
            'subsequent_attempts_by_original_position': self.subsequent_attempts_by_original_position,
        }


@dataclass(frozen=True)
class TrafficSpec:
    name: Optional[str] = dataclasses.field(compare=False, default=None)

    start_timestamp: Optional[Union[str, pd.Timestamp, datetime]] = None
    end_timestamp: Optional[Union[str, pd.Timestamp, datetime]] = None

    max_station_distance: float = float('inf')
    max_stations_pair_distance: float = float('inf')
    max_attempts: Optional[int] = 1

    minimal_complete_station_subset: bool = False
    station_subset: Optional[List[int]] = None

    subsequent_attempts_by_original_position: bool = False
    remove_redundant_station_attempts: bool = False

    def __post_init__(self):
        object.__setattr__(self, 'start_timestamp',
                           _to_datetime(self.start_timestamp))
        object.__setattr__(self, 'end_timestamp',
                           _to_datetime(self.end_timestamp))

    def copy(self, **kwargs):
        return dataclasses.replace(self, **kwargs)

    @property
    def days(self):
        return (self.end_timestamp - self.start_timestamp).days

    def to_canonical_id(self, named=False):
        out = [
            self.name if named else None,
            self.start_timestamp.date().isoformat() if self.start_timestamp is not None else None,
            'to',
            self.end_timestamp.date().isoformat() if self.end_timestamp is not None else None,
            f"{self.max_attempts}att" if self.max_attempts is not None else None,
            str(self.max_station_distance),
            str(self.max_stations_pair_distance),
            'minst' if self.minimal_complete_station_subset else None,
            'origpos' if self.subsequent_attempts_by_original_position else None,
            'stfilt' if self.remove_redundant_station_attempts else None
        ]

        return '-'.join([o for o in out if o is not None])

    def __repr__(self):
        vals = [
            self.name,
            self.to_canonical_id(named=False)
        ]

        vals = ', '.join([repr(v) for v in vals if v is not None])
        return f"TrafficSpec({vals})"

    def to_simulation_spec(self):
        return SimulationSpec(name="self",
                              max_attempts=self.max_attempts,
                              max_station_distance=self.max_station_distance,
                              max_stations_pair_distance=self.max_stations_pair_distance,
                              subsequent_attempts_by_original_position=self.subsequent_attempts_by_original_position)


DEFAULT_TRAFFIC_COLUMNS = TrafficColumns()


def build_traffic(spec: TrafficSpec, traffic_full: pd.DataFrame, charging_stations: pd.DataFrame,
                  station_distances_mtx: List[List[float]],
                  cols: TrafficColumns = DEFAULT_TRAFFIC_COLUMNS, verbose=False):
    traffic = traffic_full.copy()
    traffic.index.name = cols.index
    traffic.reset_index(cols.index, drop=False, inplace=True)

    if spec.start_timestamp is not None:
        if verbose:
            print("Dropping traffic before", spec.start_timestamp, "...")
        traffic.drop(traffic[traffic[cols.arrival] <
                             spec.start_timestamp].index, inplace=True)

    if spec.end_timestamp is not None:
        if verbose:
            print("Dropping traffic after", spec.end_timestamp, "...")
        traffic.drop(traffic[traffic[cols.arrival] >=
                             spec.end_timestamp].index, inplace=True)

    if traffic.index.duplicated().any():
        warnings.warn("Full traffic dataframe contains multiple attempts per charging request"
                      " (multiple rows with identical index).")
        if spec.max_attempts is not None and spec.max_attempts <= 1:
            traffic.drop(traffic[traffic[cols.attempt_no]
                                 != 0].index, inplace=True)
        else:
            traffic.sort_values(
                [cols.index, cols.station_distance], inplace=True)
            traffic[cols.attempt_no] = traffic.groupby(cols.index).cumcount()

    assert sum([int(b) for b in [spec.minimal_complete_station_subset,
                                 spec.station_subset is not None]]) <= 1

    if spec.minimal_complete_station_subset:
        if verbose:
            print("Computing minimal complete station subset", "...")
        traffic.set_index(cols.index, inplace=True)
        clustering.station_subset_iterative_reduction(
            1, traffic, charging_stations, inplace=True,
            max_station_distance=spec.max_station_distance,
            station_column=cols.station,
            station_distance_column=cols.station_distance,
            geo_points_cols=(cols.x, cols.y), verbose=verbose)
        traffic.reset_index(cols.index, drop=False, inplace=True)
        charging_stations = charging_stations.loc[traffic[cols.station].unique()]
    elif spec.station_subset is not None:
        if verbose:
            print("Assigning stations", "...")
        charging_stations = charging_stations.loc[spec.station_subset]
        clustering.assign_stations_to_traffic(
            traffic, charging_stations, inplace=True)


    if spec.max_attempts is None or spec.max_attempts > 1:
        if spec.subsequent_attempts_by_original_position:
            if verbose:
                print("Computing subsequent attempts based on original position", "...")
            traffic.set_index(cols.index, inplace=True)
            traffic_subseq = clustering.assign_stations_to_traffic_cross(
                traffic, charging_stations, station_column=cols.station,
                station_distance_column=cols.station_distance, geo_points_cols=(cols.x, cols.y))
            traffic_subseq.index.name = cols.index
            traffic_subseq.reset_index(cols.index, drop=False, inplace=True)
            traffic.reset_index(cols.index, drop=False, inplace=True)

            traffic_subseq.drop(traffic_subseq[traffic_subseq[cols.station_distance]
                                               > spec.max_station_distance].index, inplace=True)
            traffic_subseq.sort_values(
                [cols.index, cols.station_distance], inplace=True)

            traffic = pd.concat([traffic, traffic_subseq], ignore_index=True)
            del traffic_subseq

            traffic.drop_duplicates(
                subset=[cols.index, cols.station], keep='first', inplace=True)

            traffic[cols.attempt_no] = traffic.groupby(cols.index).cumcount()

            traffic.drop(traffic[traffic[cols.attempt_no]
                                 > spec.max_attempts].index, inplace=True)
        else:
            if verbose:
                print("Computing subsequent attempts based on station positions", "...")
            traffic.set_index(cols.index, inplace=True)
            traffic = sim.add_attempts_to_traffic(
                sim.PreprocessedTraffic.from_dataframe(
                    traffic), station_distances_mtx,
                max_attempts=(
                    spec.max_attempts if spec.max_attempts is not None else traffic[cols.station].nunique()),
                max_stations_pair_distance=spec.max_stations_pair_distance,
                station_choice_probabilistic=False)
            traffic.reset_index(cols.index, drop=False, inplace=True)

    if spec.max_station_distance < float('inf'):
        if verbose:
            print("Dropping traffic over", spec.max_station_distance,
                  "station distance", "...")
        traffic.drop(traffic[traffic[cols.station_distance]
                             > spec.max_station_distance].index, inplace=True)

    if spec.remove_redundant_station_attempts:
        if verbose:
            print("Removing redundant stations", "...")

        if (spec.max_attempts is None or spec.max_attempts > 1) and cols.attempt_no in traffic.columns and traffic[cols.attempt_no].nunique() > 1:
            traffic.sort_values([cols.arrival, cols.attempt_no], inplace=True)

            edg = list()
            traffic.groupby(cols.index)[cols.station].apply(
                lambda x: edg.extend(list(zip(x.iloc[:-1].tolist(), x.iloc[1:].tolist()))))

            predecessors = {st: set(p for p, s in edg if s == st)
                            for st in traffic[cols.station].unique()}

            nds_nonredundant = set(traffic.loc[traffic[cols.attempt_no] <= 0, cols.station].unique())\
                .union(st for st, pred in predecessors.items() if len(pred) > 1)
            nds_redundant = set(predecessors.keys()
                                ).difference(nds_nonredundant)

            if verbose:
                print("Redundant stations:", nds_redundant)

            if len(nds_redundant) > 0:
                traffic.drop(traffic[traffic[cols.station].isin(
                    nds_redundant)].index, inplace=True)
                traffic.sort_values(
                    [cols.index, cols.attempt_no], inplace=True)
                traffic[cols.attempt_no] = traffic.groupby(
                    cols.index).cumcount()

    traffic.set_index(cols.index, inplace=True)
    return traffic


def build_attempt_graph(traffic: Optional[pd.DataFrame] = None, log: Optional[pd.DataFrame] = None, directed=True, cols: TrafficColumns = DEFAULT_TRAFFIC_COLUMNS):
    if log is None and traffic is None:
        raise Exception("Either 'log' or 'traffic' must be set.")
    elif log is None:
        log = sim.traffic_to_log(traffic)
    elif traffic is not None:
        raise Exception("Either 'log' or 'traffic' must be set. Not both.")

    nod_all = set()
    for n in log[cols.station].unique():
        nod_all.add(n)

    nod_first = set()
    for n in log.loc[log[cols.attempt_no] <= 0, cols.station].unique():
        nod_first.add(n)

    nod = list()
    for n in nod_first:
        nod.append((n, {'init': True}))
    for n in nod_all.difference(nod_first):
        nod.append((n, {'init': False}))

    edg = []
    log.query('status == 1').groupby(cols.index)[cols.station].apply(
        lambda x: edg.extend(list(zip(x.iloc[:-1].tolist(), x.iloc[1:].tolist()))))
    edg = list(set(edg))

    import networkx as nx
    G = nx.DiGraph() if directed else nx.Graph()
    G.add_nodes_from(nod)
    G.add_edges_from(edg)

    return G


def render_attempt_graph(charging_stations: pd.DataFrame, *kargs, cols: TrafficColumns = DEFAULT_TRAFFIC_COLUMNS, labels=True, color_first='r', color_other: Optional[str] = None, **kwargs):
    G = build_attempt_graph(*kargs, cols=cols, **kwargs)

    pos = charging_stations.reindex(
        np.arange(max(G.nodes()) + 1))[[cols.x, cols.y]].to_numpy()

    if color_other is None:
        color_other = '#1378C7'
    if color_first is None:
        color_first = '#1378C7'

    import networkx as nx
    import matplotlib.pyplot as plt
    nx.draw_networkx(G, nodelist=[n for n, d in G.nodes(
        data=True) if d['init'] == False], node_size=300 if labels else 100, pos=pos, font_color='w', node_color=color_other, with_labels=labels)
    nx.draw_networkx_nodes(G, nodelist=[n for n, d in G.nodes(
        data=True) if d['init'] == True], node_size=300 if labels else 100, pos=pos, node_color=color_first)
