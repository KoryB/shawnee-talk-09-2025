from .constants import *
from . import memory as mem

from typing import Union, Optional

import numpy as np
import scipy.spatial.transform as sct
from numba import njit

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
    out[3, 3] = 1

    return out


def rotation_y(pitch: FLOAT_DTYPE, out: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
    out = init_out_or_pass_through(out, (4, 4))

    c, s = np.cos(pitch), np.sin(pitch)

    out[0, 0] = c
    out[0, 2] = s
    out[2, 0] = -s
    out[2, 2] = c
    
    out[1, 1] = 1
    out[3, 3] = 1

    return out


def rotation_z(yaw: FLOAT_DTYPE, out: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
    out = init_out_or_pass_through(out, (4, 4))

    c, s = np.cos(yaw), np.sin(yaw)

    out[1:3, 1:3] = np.array([c, -s, s, c], dtype=FLOAT_DTYPE).reshape(2, 2)
    out[0, 0] = 1
    out[3, 3] = 1

    return out



def rotation(roll: float, pitch: float, yaw: float, out: Optional[np.ndarray] = None) -> np.ndarray:
    # TODO: Figure out why this is causing nothing to appear if called with commented code
    # out = init_out_or_pass_through(out, (4, 4))

    # rx, rx_h = mem.get_sb(4*4, mem.SbType.FLOAT)
    # ry, ry_h = mem.get_sb(4*4, mem.SbType.FLOAT)
    # rz, rz_h = mem.get_sb(4*4, mem.SbType.FLOAT)

    # rx = rx.reshape(4, 4)
    # ry = ry.reshape(4, 4)
    # rz = rz.reshape(4, 4)
    
    # np.matmul(
    #     np.matmul(rotation_x(roll, out=rx), rotation_y(pitch, out=ry), out=out),
    #     rotation_z(yaw, out=rz), out=out)
    
    # mem.free_sb(rx_h, mem.SbType.FLOAT)
    # mem.free_sb(ry_h, mem.SbType.FLOAT)
    # mem.free_sb(rz_h, mem.SbType.FLOAT)

    return rotation_x(roll) @ rotation_y(pitch) @ rotation_z(yaw)


def projection(
        field_of_view_x: FLOAT_DTYPE,
        aspect_ratio: FLOAT_DTYPE,
        hither: FLOAT_DTYPE) -> np.ndarray:

    right = np.tan(np.deg2rad(field_of_view_x/2)) * hither
    top = right / aspect_ratio

    return np.array([
        hither/right, 0,          0,  0,
        0,            hither/top, 0,  0,
        0,            0,          -1, -2*hither,
        0,            0,          -1, 0
    ], dtype=FLOAT_DTYPE).reshape(4, 4)


def to_screen(p: np.ndarray) -> np.ndarray:
    flip = np.array([1, -1, 1])
    p = flip*p[0:3] / p[3]
    p[0:2] = p[0:2]*SCREEN_HALF_SIZE + SCREEN_HALF_SIZE

    return p


def is_in_clip(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> bool:
    return _is_in_clip(a, b, c, mem.DEPTH_BUFFER)

@njit
def _is_in_clip(a: np.ndarray, b: np.ndarray, c: np.ndarray, depth_buffer: np.ndarray) -> bool:
    return (
        -a[3] <= a[0] <= a[3] and -a[3] <= a[1] <= a[3] and a[2] > 0 and
        -b[3] <= b[0] <= b[3] and -b[3] <= b[1] <= b[3] and b[2] > 0 and
        -c[3] <= c[0] <= c[3] and -c[3] <= c[1] <= c[3] and c[2] > 0 and
        a[2] <= depth_buffer[int(np.round(a[0])), int(np.round(a[1]))] and
        b[2] <= depth_buffer[int(np.round(b[0])), int(np.round(b[1]))] and
        c[2] <= depth_buffer[int(np.round(c[0])), int(np.round(c[1]))]
    )