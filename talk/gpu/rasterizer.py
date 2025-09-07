from .constants import *
from . import memory as mem

from typing import Union, Optional

import numpy as np
import scipy.spatial.transform as sct

"""
We assume column vectors, math
"""

IDENTITY_4X4 = np.identity(4)


def init_out_or_pass_through(out: Optional[np.ndarray], shape) -> np.ndarray:
    if out is None:
        out = np.zeros(shape, dtype=FLOAT_DTYPE)

    else:
        assert out.shape == shape

    return out


def translate(v: np.ndarray, out: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
    assert v.shape == (3,)
    out = init_out_or_pass_through(out, (4, 4))

    out[:, :] = IDENTITY_4X4
    out[0:3, 3] = v

    return out


def rotation_x(roll: FLOAT_DTYPE, out: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
    out = init_out_or_pass_through(out, (4, 4))

    c, s = np.cos(roll), np.sin(roll)

    out[0:2, 0:2] = np.array([c, -s, s, c], dtype=FLOAT_DTYPE).reshape(2, 2)
    out[2, 2] = 1

    return out


def rotation_y(pitch: FLOAT_DTYPE, out: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
    out = init_out_or_pass_through(out, (4, 4))

    c, s = np.cos(pitch), np.sin(pitch)

    out[0, 0] = c
    out[0, 2] = s
    out[2, 0] = -s
    out[2, 2] = c
    
    out[1, 1] = 1

    return out


def rotation_z(yaw: FLOAT_DTYPE, out: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
    out = init_out_or_pass_through(out, (4, 4))

    c, s = np.cos(yaw), np.sin(yaw)

    out[1:3, 1:3] = np.array([c, -s, s, c], dtype=FLOAT_DTYPE).reshape(2, 2)
    out[0, 0] = 1

    return out



def rotation(roll: float, pitch: float, yaw: float, out=Optional[np.ndarray]) -> np.ndarray:
    out = init_out_or_pass_through(out)

    rx, rx_h = mem.get_sb(4*4, mem.SbType.FLOAT).reshape(4, 4)
    ry, ry_h = mem.get_sb(4*4, mem.SbType.FLOAT).reshape(4, 4)
    rz, rz_h = mem.get_sb(4*4, mem.SbType.FLOAT).reshape(4, 4)

    
    np.matmul(
        np.matmul(rotation_x(roll, out=rx), rotation_y(pitch, out=ry), out=out),
        rotation_z(yaw, out=rz), out=out)
    
    mem.free_sb(rx_h, mem.SbType.FLOAT)
    mem.free_sb(ry_h, mem.SbType.FLOAT)
    mem.free_sb(rz_h, mem.SbType.FLOAT)

    return out


def project(f: FLOAT_DTYPE) -> np.ndarray:
    return np.array([
        f, 0, SCREEN_HORIZONTAL_CENTER, 0,
        0, f, SCREEN_VERTICAL_CENTER,   0,
        0, 0, 1,                        0
    ], dtype=FLOAT_DTYPE).reshape(3, 4)