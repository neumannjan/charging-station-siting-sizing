import itertools
from typing import Callable, List, Tuple, Union

import numpy as np
import pandas as pd


def distance_squared(points1: np.ndarray, points2: np.ndarray, cross=True) -> np.ndarray:
    if cross:
        return np.sum(np.power(points1[:, np.newaxis] - points2[np.newaxis, :], 2), axis=2)
    else:
        return np.sum(np.power(points1 - points2, 2), axis=1)


def distance_nonsquared(points1: np.ndarray, points2: np.ndarray, cross=True) -> np.ndarray:
    return np.sqrt(distance_squared(points1, points2, cross))


def initialize_centroids(k: int, points: np.ndarray):
    """
    Initialize centroids for regular KMeans algorithm.
    """
    centroids = np.zeros((0, 2), dtype=float)

    rng = list(range(len(points)))

    # first point
    idx = np.random.choice(rng)

    centroids = np.vstack((centroids, points[rng[idx], :]))
    del rng[idx]

    # repeat until k
    for _ in range(k - 1):
        p = np.min(distance_squared(centroids, points[rng, :]), axis=0)
        p /= np.sum(p)

        idx = np.random.choice(range(len(rng)), p=p)
        centroids = np.vstack((centroids, points[rng[idx], :]))
        del rng[idx]

    return centroids


def assign_points_to_centroids(centroids: np.ndarray, points: np.ndarray):
    """
    Assign points to centroids for KMeans -> cluster assignment computation step.
    """
    return np.argmin(distance_squared(centroids, points), axis=0)


def centroids_from_assignment(assignment: np.ndarray, points: np.ndarray):
    """
    Compute geometric means of assignment clusters -> centroid computation step.
    """
    centroids = []

    for i in np.unique(assignment):
        centroids.append(np.mean(points[assignment == i, :], axis=0))

    return np.array(centroids, dtype=float)


def kmeans(k: int, points: np.ndarray, precision=1e-10, verbose=False):
    """
    KMeans algorithm.
    (The regular variant - computes geometric means.)
    """
    cntr = initialize_centroids(k, points)
    asgn = assign_points_to_centroids(cntr, points)

    while True:
        cntr_new = centroids_from_assignment(asgn, points)
        diff = np.sum(np.power(cntr - cntr_new, 2))

        if verbose:
            print(diff)
        if diff < precision:
            break

        cntr = cntr_new
        asgn = assign_points_to_centroids(cntr, points)

    return cntr, asgn
