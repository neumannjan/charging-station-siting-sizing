from typing import Callable, Optional, Tuple

import geopandas as gpd
import numpy as numpy
import pandas as pd

CRS_WGS84 = "EPSG:4326"
CRS_LONGITUDE_LATITUDE = CRS_WGS84
CRS_UTM_METRIC_CZECH_REPUBLIC = "EPSG:32633"


def convert_crs_points_series(x: pd.Series, y: pd.Series,
                              input_crs: str, output_crs: str):
    """Convert a series of points between two CRS, where the points are represented
    as a pair of two Pandas Series (first for x values, second for y values).

    :param x:  Pandas Series of X values to convert from
    :type  x:  pd.Series
    :param y:  Pandas Series of Y values to convert from
    :type  y:  pd.Series
    :param input_crs: Input CRS string
    :type  input_crs: str
    :param output_crs: Output CRS string
    :type  output_crs: str

    :return:  The points in the output CRS as a tuple of two Pandas Series
              (first for x values, second for y values).
    :rtype :  Tuple[pd.Series, pd.Series]
    """
    assert (x.index == y.index).all()
    g_points = gpd.GeoDataFrame(index=x.index, geometry=gpd.points_from_xy(
        x, y), crs=input_crs)['geometry']
    g_points = g_points.to_crs(output_crs)
    return g_points.x, g_points.y


def convert_crs_points_dataframe(df: pd.DataFrame,
                                 input_crs: str, output_crs: str,
                                 input_cols=('lon', 'lat'), output_cols=('x', 'y'), copy=False):
    """Convert a series of points between two CRS, where the points are represented
    as two columns in a Pandas DataFrame (one column for X values, one for Y values).

    :param df:          The input DataFrame
    :type  df:          pd.DataFrame
    :param input_crs:   Input CRS string
    :type  input_crs:   str
    :param output_crs:  Output CRS string
    :type  output_crs:  str
    :param input_cols:  A tuple of input column names
    :type  input_cols:  Tuple[str, str]
    :param output_cols: A tuple of output column names
    :type  output_cols: Tuple[str, str]
    :param copy:        If false, the dataframe is modified in place. If true, it is copied.
    :type  copy:        bool

    :return:  The updated DataFrame, if copy == True. Otherwise, no return value.
    :rtype :  Optional[pd.DataFrame]
    """

    if copy:
        df = df.copy()

    x, y = convert_crs_points_series(
        x=df[input_cols[0]], y=df[input_cols[1]], input_crs=input_crs, output_crs=output_crs)
    df[output_cols[0]] = x
    df[output_cols[1]] = y

    if copy:
        return df
