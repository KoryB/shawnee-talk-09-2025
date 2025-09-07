from .constants import *
from . import memory as mem

from typing import Tuple, Union


def get_grid(min: np.ndarray, max: np.ndarray) -> np.ndarray:
    return np.mgrid[min[0]:max[0], min[1]:max[1]].reshape(2,-1).T


def linspace_int(
        a: INTEGER_DTYPE, 
        b: INTEGER_DTYPE, 
        npoints: UNSIGNED_INTEGER_DTYPE, 
        out: Union[np.ndarray, None] = None
    ) -> Union[np.ndarray, UNSIGNED_INTEGER_DTYPE]:
    
    if out is None:
        return np.linspace(a, b, npoints, dtype=INTEGER_DTYPE)
    
    assert out.size >= npoints

    buff, buff_h = mem.get_sb(npoints, mem.SbType.FLOAT)

    np.divide(mem.SEQUENCE_UINT[:npoints], npoints, out=buff, dtype=FLOAT_DTYPE)  # interpolation
    buff *= (b - a)  # Scale
    buff += a  # min value
    
    out[:npoints] = buff  # copy to output, this will truncate integers

    mem.free_sb(buff_h, mem.SbType.FLOAT)

    return npoints


def interpolate_y(a: np.ndarray, b: np.ndarray, out: np.ndarray) -> np.ndarray:
    npoints = b[0] - a[0] + 1

    return linspace_int(a[1], b[1], npoints, out=out)


def interpolate_x(a: np.ndarray, b: np.ndarray, out: np.ndarray) -> np.ndarray:
    npoints = b[1] - a[1] + 1

    return linspace_int(a[0], b[0], npoints, out=out)