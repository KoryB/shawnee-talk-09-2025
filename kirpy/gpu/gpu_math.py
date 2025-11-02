from .constants import *
from . import memory as mem

import numpy as np


def interpolate_y(a: np.ndarray, b: np.ndarray, out: np.ndarray=None) -> np.ndarray:
    npoints = b[0] - a[0] + 1

    return np.linspace(a[1], b[1], npoints, dtype=INTEGER_DTYPE)


def interpolate_x(a: np.ndarray, b: np.ndarray, out: np.ndarray=None) -> np.ndarray:
    npoints = b[1] - a[1] + 1

    return np.linspace(a[0], b[0], npoints, dtype=INTEGER_DTYPE)