from dataclasses import dataclass
from typing import Tuple, Union
from contextlib import contextmanager

import numpy as np
import pygame
from numba import njit, prange
from enum import Enum


"""
Everything is assumed to be single threaded (yay)
"""


class ScratchBufferType(Enum):
    INT = 0
    UINT = 1
    FLOAT = 2


SURFACE_MODE = "RGBA"
OUTPUT_WIDTH = 320
OUTPUT_HEIGHT = 240
NUM_SCRATCH_BUFFERS = 32

INTEGER_DTYPE = np.int16
UNSIGNED_INTEGER_DTYPE = np.uint16
FLOAT_DTYPE = np.float32
MAX_INT = np.iinfo(INTEGER_DTYPE).max
MAX_UINT = np.iinfo(UNSIGNED_INTEGER_DTYPE).max
SEQUENCE_INT = np.arange(MAX_INT, dtype=INTEGER_DTYPE)
SEQUENCE_UINT = np.arange(MAX_UINT, dtype=UNSIGNED_INTEGER_DTYPE)
BUFFER_MGRID_Y, BUFFER_MGRID_X = np.mgrid[:OUTPUT_HEIGHT, :OUTPUT_WIDTH].astype(UNSIGNED_INTEGER_DTYPE)

SCRATCH_BUFFER_SIZE = MAX_UINT

# Could optimize to work as an object pool type thing
SCRATCH_BUFFER_INT = np.zeros((NUM_SCRATCH_BUFFERS, SCRATCH_BUFFER_SIZE), dtype=INTEGER_DTYPE)
SCRATCH_BUFFER_UINT = np.zeros((NUM_SCRATCH_BUFFERS, SCRATCH_BUFFER_SIZE), dtype=UNSIGNED_INTEGER_DTYPE)
SCRATCH_BUFFER_FLOAT = np.zeros((NUM_SCRATCH_BUFFERS, SCRATCH_BUFFER_SIZE), dtype=FLOAT_DTYPE)

SCRATCH_BUFFER_INT_IN_USE = np.array([False] * NUM_SCRATCH_BUFFERS, dtype=bool)
SCRATCH_BUFFER_UINT_IN_USE = np.array([False] * NUM_SCRATCH_BUFFERS, dtype=bool)
SCRATCH_BUFFER_FLOAT_IN_USE = np.array([False] * NUM_SCRATCH_BUFFERS, dtype=bool)

SCRATCH_BUFFERS = {
    ScratchBufferType.INT: SCRATCH_BUFFER_INT,
    ScratchBufferType.UINT: SCRATCH_BUFFER_UINT,
    ScratchBufferType.FLOAT: SCRATCH_BUFFER_FLOAT,
}

SCRATCH_BUFFER_IN_USES = {
    ScratchBufferType.INT: SCRATCH_BUFFER_INT_IN_USE,
    ScratchBufferType.UINT: SCRATCH_BUFFER_UINT_IN_USE,
    ScratchBufferType.FLOAT: SCRATCH_BUFFER_FLOAT_IN_USE,
}


class Color:
    def __init__(self, r: np.uint8, g: np.uint8, b: np.uint8, a: np.uint8 = 0xFF):
        self.r = r
        self.g = g
        self.b = b
        self.a = a

        self.array = np.array([r, g, b, a], dtype=np.uint8)


DEFAULT_COLOR = Color(255, 0, 255)


class Buffer:
    def __init__(self, array: np.ndarray):
        self.array = array
        self.mode = SURFACE_MODE


    def get_surface(self):
        h, w, _ = self.array.shape

        surf = pygame.image.frombuffer(self.array.data, (w, h), SURFACE_MODE)

        return surf
    

def get_sb(size: UNSIGNED_INTEGER_DTYPE, type: ScratchBufferType) -> Union[np.ndarray, UNSIGNED_INTEGER_DTYPE]:
    sbs = SCRATCH_BUFFERS[type]
    in_use = SCRATCH_BUFFER_IN_USES[type]

    assert ~np.all(in_use)

    handle = np.argmin(in_use)
    in_use[handle] = True

    return sbs[handle, :size], handle


def free_sb(handle: UNSIGNED_INTEGER_DTYPE, type: ScratchBufferType):
    assert SCRATCH_BUFFER_IN_USES[type][handle]
    SCRATCH_BUFFER_IN_USES[type][handle] = False


def get_buffer(width: int, height: int, color: Color = Color(0xFF, 0x00, 0xFF)) -> Buffer:
    raw = np.tile(color.array, (height, width, 1))
    buff = Buffer(raw)

    return buff


def clear(buff: Buffer, color: Color):
    buff.array[:, :, :] = color.array


def draw_rect(buff: Buffer, x: int, y: int, w: int, h: int, color: Color):
    buff.array[y:(y+h), x:(x+w), :] = color.array


def get_aabb(*points: np.ndarray):
    min = np.min(points, axis=0)
    max = np.max(points, axis=0)

    return min, max


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
    
    # assert out.size == npoints, temp is not None, a != b

    # with sb_float(npoints) as buff:

    buff, buff_h = get_sb(npoints, ScratchBufferType.FLOAT)

    np.divide(SEQUENCE_UINT[:npoints], npoints, out=buff, dtype=FLOAT_DTYPE)  # interpolation
    buff *= (b - a)  # Scale
    buff += a  # min value
    
    out[:npoints] = buff  # copy to output, this will truncate integers

    free_sb(buff_h, ScratchBufferType.FLOAT)

    return npoints


def interpolate_y(a: np.ndarray, b: np.ndarray, out: np.ndarray) -> np.ndarray:
    npoints = b[0] - a[0] + 1

    return linspace_int(a[1], b[1], npoints, out=out)


def interpolate_x(a: np.ndarray, b: np.ndarray, out: np.ndarray) -> np.ndarray:
    npoints = b[1] - a[1] + 1

    return linspace_int(a[0], b[0], npoints, out=out)


def get_line(a: np.ndarray, b: np.ndarray, out_x: np.ndarray, out_y: np.ndarray):
    npoints = np.max(np.array([np.abs(b[0] - a[0]) + 1, np.abs(b[1] - a[1]) + 1]))
    linspace_int(a[0], b[0], npoints, out=out_x)
    linspace_int(a[1], b[1], npoints, out=out_y)

    return npoints


def draw_line(array: np.ndarray, a: np.ndarray, b: np.ndarray, color: np.ndarray):
    out_x, out_x_h = get_sb(SCRATCH_BUFFER_SIZE, ScratchBufferType.INT)
    out_y, out_y_h = get_sb(SCRATCH_BUFFER_SIZE, ScratchBufferType.INT)

    npoints = get_line(a, b, out_x, out_y)
    array[out_y[:npoints], out_x[:npoints], :] = color

    free_sb(out_y_h)
    free_sb(out_x_h)


@njit
def triangle_blit(buff: np.ndarray, x_left: np.ndarray, x_right: np.ndarray, y0: np.int16, y1: np.int16, color: np.ndarray):
    for y in range(y0, y1 + 1):
        i = y - y0
        buff[y, x_left[i]:(x_right[i]+1), :] = color


def draw_triangle(buff: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray, color: np.ndarray):
    # Sort point y values
    if b[1] < a[1]:
        a, b = b, a

    if c[1] < a[1]:
        a, c = c, a

    if c[1] < b[1]:
        b, c = c, b

    xab_full, xab_full_h = get_sb(SCRATCH_BUFFER_SIZE, ScratchBufferType.INT)
    xbc_full, xbc_full_h = get_sb(SCRATCH_BUFFER_SIZE, ScratchBufferType.INT)
    xac_full, xac_full_h = get_sb(SCRATCH_BUFFER_SIZE, ScratchBufferType.INT)
    xabc_full, xabc_full_h = get_sb(SCRATCH_BUFFER_SIZE, ScratchBufferType.INT)

    npoints_xab = interpolate_x(a, b, out=xab_full)
    npoints_xbc = interpolate_x(b, c, out=xbc_full)
    npoints_xac = interpolate_x(a, c, out=xac_full)

    xab = xab_full[:npoints_xab]
    xbc = xbc_full[:npoints_xbc]
    xac = xac_full[:npoints_xac]
    xabc = xabc_full[:npoints_xac]

    np.concatenate([xab[:-1], xbc], out=xabc)

    midpoint = np.floor_divide(xabc.size, 2, dtype=UNSIGNED_INTEGER_DTYPE)

    if xac[midpoint] < xabc[midpoint]:
        x_left = xac
        x_right = xabc

    else:
        x_left = xabc
        x_right = xac

    """
    TODO KB:
        This ended up being significantly slower, numpy mask blit operations are slow apparently
    """
    # ys = slice(a[1], c[1]+1)
    # x_grid = (x_left[:, np.newaxis] <= BUFFER_MGRID_X[ys, 0:200]) & (BUFFER_MGRID_X[ys, 0:200] <= x_right[:, np.newaxis])
    # buff[ys, 0:200][x_grid] = color

    triangle_blit(buff, x_left, x_right, a[1], c[1], color)

    free_sb(xabc_full_h, ScratchBufferType.INT)
    free_sb(xac_full_h, ScratchBufferType.INT)
    free_sb(xbc_full_h, ScratchBufferType.INT)
    free_sb(xab_full_h, ScratchBufferType.INT)

    


def get_triangle_points_bary(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    # I'll need to do bounds detection

    aabb_min, aabb_max = get_aabb(a, b, c)

    if np.array_equal(aabb_min, aabb_max):
        return

    # https://stackoverflow.com/questions/32208359/is-there-a-multi-dimensional-version-of-arange-linspace-in-numpy
    points = get_grid(aabb_min, aabb_max)

    v0 = b - a
    v1 = c - a
    v2_points = points - a

    d00 = np.dot(v0, v0)
    d01 = np.dot(v0, v1)
    d11 = np.dot(v1, v1)
    d20 = np.dot(v2_points, v0)
    d21 = np.dot(v2_points, v1)
    
    denom = d00 * d11 - d01 * d01;

    v = (d11 * d20 - d01 * d21) / denom;
    w = (d00 * d21 - d01 * d20) / denom;
    u = 1.0 - v - w;

    mask = (v > 0) & (w > 0) & (u > 0) & (v + w + u <= 1.0)

    return points[mask]


# https://gamedev.stackexchange.com/questions/23743/whats-the-most-efficient-way-to-find-barycentric-coordinates
# Currently only accepts ints, operates directly on the buffer
# def draw_triangle(buff: Buffer, a: np.ndarray, b: np.ndarray, c: np.ndarray, color: Color):
#     triangle_points = get_triangle_points(a, b, c)

#     buff.array[triangle_points[:, 1], triangle_points[:, 0], :] = color.as_numpy()[np.newaxis, np.newaxis, :]

