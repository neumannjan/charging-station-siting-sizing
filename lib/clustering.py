from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

import lib.kmeans as kmeans
import lib.utils as utils


def assign_stations_to_traffic(traffic: pd.DataFrame, charging_stations: pd.DataFrame, inplace=False,
                               station_column: str = 'station', station_distance_column: str = 'station_distance',
                               geo_points_cols: Tuple[str, str] = ('x', 'y')):
    points = traffic[list(geo_points_cols)].to_numpy()

    dists = kmeans.distance_nonsquared(
        points, charging_stations[list(geo_points_cols)].to_numpy())
    dists_idx = np.argmin(dists, axis=1)

    if not inplace:
        traffic = traffic.copy()

    try:
        traffic[station_column] = charging_stations.index[dists_idx]
        traffic[station_distance_column] = np.min(dists, axis=1)
    except Exception as e:
        print(dists.shape, dists_idx.shape)
        raise Exception() from e

    if not inplace:
        return traffic


def assign_stations_to_traffic_cross(traffic: pd.DataFrame, charging_stations: pd.DataFrame,
                                     station_column: str = 'station', station_distance_column: str = 'station_distance',
                                     geo_points_cols: Tuple[str, str] = ('x', 'y')):
    points = traffic[list(geo_points_cols)].to_numpy()

    dists = kmeans.distance_nonsquared(
        points, charging_stations[list(geo_points_cols)].to_numpy())

    traffic = traffic.copy()
    traffic.reset_index(inplace=True)
    cols = [c for c in traffic.columns if c not in [
        station_column, station_distance_column]]

    traffic.set_index(cols, inplace=True)
    traffic[station_column] = [list(charging_stations.index) for _ in range(len(traffic))]
    traffic[station_distance_column] = list(dists)
    traffic = traffic.apply(pd.Series.explode)
    traffic.reset_index(inplace=True)
    traffic.set_index(cols[0], inplace=True)

    return traffic


def stations_from_clusters(traffic: pd.DataFrame, assignment: Union[np.ndarray, pd.Series, List[int]],
                           charging_stations: pd.DataFrame,
                           geo_points_cols: Tuple[str, str] = ('x', 'y')):
    """
    Find a station subset of k stations from an assignment of charging data to k clusters.
    The station subset is selected in order to reflect the assignment of charging data the best.

    :param traffic:  Traffic dataframe
    :param assignment: Assignment of traffic items to clusters (indexed by traffic index if pd.Series; indexed by position otherwise.)
    :param charging_stations: Charging stations dataframe
    :param geo_points_cols: DataFrame columns for X and Y axes (must be metric; i.e. euclidean distance computation must be valid)

    :return: Subset of station indices.
    :rtype:  List[int]
    """

    station_points = charging_stations[list(geo_points_cols)].to_numpy()

    result_stations_new = []
    for i in np.unique(assignment):
        points_this = traffic.loc[assignment ==
                                  i][list(geo_points_cols)].to_numpy()

        result_stations_new.append(np.argmin(np.max(
            kmeans.distance_nonsquared(station_points, points_this), axis=1), axis=0))

    return charging_stations.index[result_stations_new].tolist()


def station_subset_k_from_init_clusters(traffic: pd.DataFrame, assignment: Union[np.ndarray, pd.Series, List[int]],
                                        charging_stations: pd.DataFrame, inplace=False,
                                        geo_points_cols: Tuple[str, str] = ('x', 'y'), verbose=False, **kwargs):
    """
    Find a station subset of k stations from an initial assignment of charging data to k clusters
    using an iterative approach, where the assignment is updated as the list of stations changes.

    The approach is similar to conventional KMeans:

    1. Finds optimal stations for clusters (picks station with min(max(distance to point)) )
    2. Recomputes cluster assignment
    3. Iterates until the station assignment does not change


    :param traffic:  Traffic dataframe
    :param assignment: Assignment of traffic items to clusters (indexed by traffic index if pd.Series; indexed by position otherwise.)
    :param charging_stations: Charging stations dataframe
    :param verbose: Outputs debug information if true
    :param geo_points_cols: DataFrame columns for X and Y axes (must be metric; i.e. euclidean distance computation must be valid)

    :return:  Updated traffic dataframe if inplace=False. Otherwise, does not return.
    :rtype :  pd.DataFrame
    """

    points = traffic[list(geo_points_cols)].to_numpy()

    result_stations = np.array([], dtype=int)

    while True:
        # update stations from assignment (cluster values not used)
        result_stations_new = stations_from_clusters(
            traffic, assignment, charging_stations, geo_points_cols)

        if len(result_stations) == len(result_stations_new):
            dist = np.sum(result_stations != result_stations_new)
            if verbose:
                print(dist)

            if dist == 0:
                break

        # update assignment from stations
        assignment = kmeans.assign_points_to_centroids(
            charging_stations.loc[result_stations_new][list(geo_points_cols)].to_numpy(), points)

        result_stations = result_stations_new

    return assign_stations_to_traffic(traffic, charging_stations[result_stations], inplace=inplace, geo_points_cols=geo_points_cols, **kwargs)


def station_subset_from_kmeans(k: int, traffic: pd.DataFrame, charging_stations: pd.DataFrame, inplace=False,
                               geo_points_cols: Tuple[str, str] = ('x', 'y'), verbose=False, kmeans_kwargs=dict(), **kwargs):
    """
    Computes assignment of traffic to k stations (given a set of allowed station locations)
    using an approach based on KMeans:

    1. Finds clustering using regular KMeans (see `kmeans` function)
    2. Using the clustering from step 1, iteratively finds the k stations and the updated clustering
       (see `stational_kmeans_from_init_assignment` function)
    3. Adds/updates the DataFrame 'station' and 'station_distance' columns, to represent the
       assigned station and the distance of the point from it.

    Updates the DataFrame in place (if inplace==True). Otherwise, returns updated DataFrame.
    """

    points = traffic[list(geo_points_cols)].to_numpy()

    _, asgn = kmeans.kmeans(k, points, verbose=verbose, **kmeans_kwargs)

    return station_subset_k_from_init_clusters(
        traffic, asgn, charging_stations, verbose=verbose, geo_points_cols=geo_points_cols, **kwargs)


def station_subset_iterative_reduction(k: int, traffic: pd.DataFrame, charging_stations: pd.DataFrame, inplace=False,
                                       initial_assignment=False,
                                       max_station_distance=float('inf'),
                                       station_column: str = 'station', station_distance_column: str = 'station_distance',
                                       geo_points_cols: Tuple[str, str] = ('x', 'y'), verbose=False, **kwargs):
    if not inplace:
        traffic = traffic.copy()

    inner_kwargs: Dict = dict(station_column=station_column,
                              station_distance_column=station_distance_column, geo_points_cols=geo_points_cols)

    if initial_assignment:
        assign_stations_to_traffic(
            traffic, charging_stations, inplace=True, **inner_kwargs)

    stations = set(traffic[station_column].unique())

    if len(stations) <= 1:
        raise Exception("Cannot reduce.")

    def station_distance_diff(df):
        found_max = assign_stations_to_traffic(
            df, charging_stations.loc[list(stations.difference([df.name]))], **inner_kwargs)[station_distance_column].max()

        if (found_max > max_station_distance).any():
            result = float('inf')
        else:
            result = found_max - df[station_distance_column].max()

        if verbose >= 2:
            print(df.name, "->", result)

        return result

    for _ in utils.progressify(list(range(len(stations)-1, k-1, -1)), "Iterations", enabled=not not verbose):
        trf = traffic[[station_column,
                       station_distance_column, *geo_points_cols]]
        trf = trf.groupby(station_column).apply(station_distance_diff)
        to_remove = trf.idxmin()

        if trf.loc[to_remove] > max_station_distance:
            # quitting early
            return traffic

        stations.remove(to_remove)
        assign_stations_to_traffic(
            traffic, charging_stations.loc[list(stations)], inplace=True, **inner_kwargs)

    if not inplace:
        return traffic


def station_subset_minimal_complete_via(station_subset_k_func: Callable, traffic: pd.DataFrame, charging_stations: pd.DataFrame,
                                        max_station_distance: float, reuse_result: bool = False,
                                        station_column: str = 'station', station_distance_column: str = 'station_distance',
                                        verbose=False, **func_kwargs):
    if 'inplace' in func_kwargs:
        raise Exception('inplace parameter not supported.')

    all_kwargs = dict(traffic=traffic, charging_stations=charging_stations, max_station_distance=max_station_distance,
                      station_column=station_column, station_distance_column=station_distance_column,
                      verbose=verbose, **func_kwargs)

    next_k = traffic[station_column].nunique() - 1

    for next_k in utils.progressify(range(next_k, 0, -1), "Iterations", enabled=not not verbose):
        traffic_next = station_subset_k_func(k=next_k, **all_kwargs)

        if traffic_next[station_column].nunique() > next_k:
            # inner function probably returned original dataframe
            # we therefore quit early and return previous
            return traffic

        if (traffic_next[station_distance_column] > max_station_distance).sum() > 0:
            return traffic

        traffic = traffic_next

        if reuse_result:
            all_kwargs['traffic'] = traffic
