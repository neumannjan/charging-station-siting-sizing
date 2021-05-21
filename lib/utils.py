import copy
import functools
import itertools
import multiprocessing
from concurrent import futures
from typing import (Any, Callable, Dict, Hashable, Iterable, List, Optional,
                    Tuple, TypeVar, Union)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import cPickle as pickle
except:
    import pickle

import gzip

_S = TypeVar('_S')
_T = TypeVar('_T')
_U = TypeVar('_U')


class Cache:
    def __init__(self):
        self.cache: Dict = dict()

    def wrap(self, key: Hashable, provider: Callable[[], _T]) -> _T:
        if not isinstance(key, Hashable):
            # bypass caching
            return provider()

        value = self.cache.get(key, None)
        if value is not None:
            return value

        value = provider()
        self.cache[key] = value
        return value

    def force(self, key: Hashable, value: _T):
        self.cache[key] = value

    def drop(self, key: Hashable):
        self.cache.pop(key, None)

    def contains(self, key: Hashable) -> bool:
        return key in self.cache

    def clear(self):
        self.cache.clear()


def progressify(iterable: Iterable[_T], desc: str, enabled=True, **kwargs) -> Iterable[_T]:
    """Add a tqdm progress bar, without making it a project dependency.
    The progress bar can be disabled completely if `enabled` is set to False.
    Additional kwargs are passed to tqdm.

    :param iterable:  The iterable to wrap with a progress bar
    :param desc: Progress bar description
    :type  desc: str
    :param enabled: If false, the progress bar is skipped completely
    :type  enabled: bool

    :return:  The original iterable wrapped with a tqdm progress bar
              (or the completely original iterable instance if tqdm unavailable or progress bar disabled)
    """

    if enabled:
        try:
            from tqdm.auto import tqdm
            return tqdm(iterable=iterable, desc=desc, leave=False, **kwargs)
        except ImportError:
            return iterable
    else:
        return iterable


def kwargs_product(**constructor_kwargs) -> Iterable[Dict[str, Any]]:
    keys = constructor_kwargs.keys()
    vals = [v if isinstance(v, Iterable) and not isinstance(v, str) and not isinstance(v, pd.DataFrame) and not isinstance(v, np.ndarray)
            else [v] for v in constructor_kwargs.values()]

    for params in itertools.product(*vals):
        yield dict(zip(keys, params))


def factory(constructor: Callable[..., _T], times: int = 1, **constructor_kwargs) -> Iterable[_T]:
    """Call a function multiple times with every possible combination of kwargs.
    Each keyword argument to be passed to the function is passed to factory as a list of possible
    values to be passed to the function. All combinations of possible argument values are then used
    as arguments for the function.

    :param constructor:  The function to execute
    :param times: How many times to call the function with one particular argument combination (default: 1)
    :type  times: int


    :return:  Iterable of all function return values
    """

    for params in kwargs_product(**constructor_kwargs):
        for _ in range(times):
            yield constructor(**params)


def kwargs_subsets_product(
        *constructor_kwargs_subsets: List[Dict[str, Any]],
        reductions: Dict[str, Callable[[Any, Any], Any]] = dict()) -> Iterable[Dict[str, Any]]:

    def reduction_func(a: dict, b: dict):
        result = {**a, **b}

        for key, redct in reductions.items():
            if key in a and key in b:
                result[key] = redct(a[key], b[key])
        return result

    for ds in itertools.product(*constructor_kwargs_subsets):
        yield functools.reduce(reduction_func, ds, dict())


def factory_subsets(
        constructor: Callable[..., _T],
        *constructor_kwargs_subsets,
        reductions: Dict[str, Callable[[Any, Any], Any]] = dict(),
        times: int = 1) -> Iterable[_T]:
    for params in kwargs_subsets_product(*constructor_kwargs_subsets, reductions=reductions):
        for _ in range(times):
            yield constructor(**params)


def normalize_dataframe(df: pd.DataFrame, int_fields: Iterable[str] = set(),
                        cat_fields: Iterable[str] = set(),
                        timestamp_fields: Iterable[str] = set(),
                        obj_fields: Iterable[str] = set(),
                        int_na=None, inplace=False):

    if not inplace:
        df = df.copy()

    all_fields = set(df.select_dtypes(include=object).columns)
    int_fields = set(int_fields)
    cat_fields = set(cat_fields)
    timestamp_fields = set(timestamp_fields)
    obj_fields = set(obj_fields)

    for f in int_fields:
        df[f] = df[f].fillna(int_na).astype(int)
    for f in cat_fields:
        df[f] = df[f].astype('category')
    for f in timestamp_fields:
        df[f] = pd.to_datetime(df[f])
    for f in obj_fields:
        df[f] = df[f].astype('object')
    for f in all_fields.difference(int_fields | cat_fields | timestamp_fields | obj_fields):
        df[f] = df[f].astype('string')

    if not inplace:
        return df


def convert_timezone(df: pd.DataFrame, columns: Iterable[str], timezone: str, inplace=False):
    if not inplace:
        df = df.copy()

    for c in columns:
        df[c] = df[c].dt.tz_convert(timezone)

    if not inplace:
        return df


def gzip_pickle_dump(obj, file_path: str):
    if obj is None:
        return

    with gzip.GzipFile(file_path, 'w') as f:
        pickle.dump(obj, f)


def gzip_pickle_load(file_path: str):
    with gzip.GzipFile(file_path, 'r') as f:
        return pickle.load(f)


def display(*kargs, **kwargs):
    try:
        from IPython.display import display
        display(*kargs, **kwargs)
    except:
        print(*kargs, **kwargs)


def iter_values(iterable: Union[Iterable[_T], Dict[Any, _T]]) -> Iterable[_T]:
    if isinstance(iterable, dict):
        return iterable.values()

    return iterable


def iter_items(iterable: Union[Iterable[_T], Dict[_U, _T]], key_provider: Callable[[_T], _U]) -> Iterable[Tuple[_U, _T]]:
    if isinstance(iterable, dict):
        yield from iterable.items()
    else:
        for v in iterable:
            yield (key_provider(v), v)


def iter_keys(iterable: Union[Iterable[_T], Dict[_U, _T]], key_provider: Callable[[_T], _U]) -> Iterable[_U]:
    if isinstance(iterable, dict):
        yield from iterable.keys()
    else:
        for v in iterable:
            yield key_provider(v)


def call_with_kargs(func: Callable[..., _T], kargs: List[Any]) -> _T:
    return func(*kargs)


def call_with_kwargs(func: Callable[..., _T], kwargs: Dict[str, Any]) -> _T:
    return func(**kwargs)


def call_with(func: Callable[..., _T], args: Tuple[List[Any], Dict[str, Any]]) -> _T:
    kargs, kwargs = args
    return func(*kargs, **kwargs)


def map_value(value_map_func: Callable[[_U], _T], keyval_pair: Tuple[_S, _U]) -> Tuple[_S, _T]:
    key, val = keyval_pair
    return key, value_map_func(val)


def int_list_printable(vals: List[int]) -> str:
    results = []

    for v in vals:
        if len(results) >= 1 and results[-1][-1] == v - 1:
            results[-1].append(v)
        else:
            results.append([v])

    results = [str(r[0]) if len(
        r) == 1 else f"{r[0]}-{r[-1]}" for r in results]
    return ', '.join(results)


def chunks(lst: _T, n: int) -> List[_T]:
    """Yield n successive chunks from lst."""
    is_dict = isinstance(lst, dict)

    if is_dict:
        lst = list(lst.items())

    chunksize = (len(lst) // n) + (0 if len(lst) % n == 0 else 1)

    i = 0
    while True:
        val = lst[i:i+chunksize]
        if len(val) == 0:
            break

        if is_dict:
            yield {k: v for k, v in val}
        else:
            yield val
        i += chunksize


def electrifiablePercentage(carids, sumLensTripsPerDayRaw, maxDaySum):
    """
    returns dictionary {id:number of noneectrifiable days car was used}, dictionary {id:number of electrifiable days}
    and list of percentage of electrifiable days of every car
    :param carids: list of ids of cars
    :param sumLensTripsPerDayRaw: list of tuples (id, sum of lengths of trips, date, sum of trips)
    :param maxDaySum: max day sum length for electrification
    :return: {int:int}, {int:int}, [int]
    """
    nonElectrifiableDay = {}
    goodElectrifiableDay = {}
    for id in carids:
        nonElectrifiableDay[id] = 0
        goodElectrifiableDay[id] = 0
    for day in sumLensTripsPerDayRaw:
        if day[1] > maxDaySum:
            nonElectrifiableDay[day[0]] = nonElectrifiableDay[day[0]] + 1
        else:
            goodElectrifiableDay[day[0]] = goodElectrifiableDay[day[0]] + 1
    electrifiableDayPercentage = []
    for id in carids:
        if goodElectrifiableDay[id] == 0:
            electrifiableDayPercentage.append(0)
        else:
            pom = 100 * \
                (goodElectrifiableDay[id] /
                 (goodElectrifiableDay[id] + nonElectrifiableDay[id]))
            electrifiableDayPercentage.append(pom)
    return nonElectrifiableDay, goodElectrifiableDay, electrifiableDayPercentage


def electrifiablePluginReady(pluginElectrifiableDaysPercentage, electrifiableDayPercentage, electroMinPerc,
                             pluginMinPerc):
    """
    returns number of electrifiable cars and number of plugin ready (possibly substitued for plug-in hybrid) cars
    :param pluginElectrifiableDaysPercentage: list of percentages (of days hybrid can substitue current vehicle without limitations)
    :param electrifiableDayPercentage: list of percentages (of days ev can substitue current vehicle without limitations)
    :param electroMinPerc: minimal electrifiableDayPercentage to be accepted
    :param pluginMinPerc: minimal pluginElectrifiableDayPercentage to be accepted
    :return: int, int
    """
    pluginReady = 0
    electrifiable = 0
    for i in range(0, len(pluginElectrifiableDaysPercentage)):
        if electrifiableDayPercentage[i] > pluginMinPerc and electrifiableDayPercentage[i] < electroMinPerc:
            if pluginElectrifiableDaysPercentage[i] > pluginMinPerc:
                pluginReady += 1
        elif electrifiableDayPercentage[i] >= electroMinPerc:
            electrifiable += 1
    return electrifiable, pluginReady


def compute_and_print_demand_stats(demand_traffic):
    # extend dataframe by computed values
    traffic = demand_traffic.copy()

    traffic.loc[:, 'duration'] = traffic['departure'] - traffic['arrival']
    traffic.loc[:, 'duration_hours'] = traffic['duration'].dt.total_seconds() / \
        60 / 60
    cars_count = len(traffic['car_id'].unique())
    # print(traffic.describe())

    car_count = len(traffic['car_id'].unique())
    overall_duration = traffic['duration'].sum()

    duration = traffic['departure'] - traffic['arrival']
    duration_total_h = duration.sum().total_seconds() / 60 / 60

    days_total = (traffic['departure'].max(
    ) - traffic['departure'].min()).total_seconds() / 60 / 60 / 24
    charging_h_per_day_and_car = duration_total_h / days_total / car_count

    print(f"Počet vozidel: {car_count}")
    print(f"Počet nabíjení: {len(traffic)}")
    print(f"Poptávka za vozidlo a den: {charging_h_per_day_and_car:.3f} hodin")
    print(
        f"Počet dní: {days_total:.1f}, od {traffic['departure'].min()} do {traffic['departure'].max()}")
    traffic['duration_hours'].hist(bins=50)
    plt.title("Histogram")
    plt.xlabel('Charging duration [h]')
    plt.ylabel('Vehicle count')
    return traffic
