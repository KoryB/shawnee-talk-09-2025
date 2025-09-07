from .constants import *
from . import gpu_math as gpum
from . import memory as mem
from .screen import Screen, Color

import numpy as np
from numba import njit, prange



def clear(buff: Screen, color: Color):
    buff.array[:, :, :] = color.array


def draw_rect(buff: Screen, x: int, y: int, w: int, h: int, color: Color):
    buff.array[y:(y+h), x:(x+w), :] = color.array


def get_line(a: np.ndarray, b: np.ndarray, out_x: np.ndarray, out_y: np.ndarray):
    npoints = np.max(np.array([np.abs(b[0] - a[0]) + 1, np.abs(b[1] - a[1]) + 1]))
    gpum.linspace_int(a[0], b[0], npoints, out=out_x)
    gpum.linspace_int(a[1], b[1], npoints, out=out_y)

    return npoints


def draw_line(array: np.ndarray, a: np.ndarray, b: np.ndarray, color: np.ndarray):
    out_x, out_x_h = mem.mem.get_sb(SCRATCH_BUFFER_SIZE, mem.SbType.INT)
    out_y, out_y_h = mem.mem.get_sb(SCRATCH_BUFFER_SIZE, mem.SbType.INT)

    npoints = get_line(a, b, out_x, out_y)
    array[out_y[:npoints], out_x[:npoints], :] = color

    mem.mem.free_sb(out_y_h)
    mem.mem.free_sb(out_x_h)


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

    xab_full, xab_full_h = mem.get_sb(SCRATCH_BUFFER_SIZE, mem.SbType.INT)
    xbc_full, xbc_full_h = mem.get_sb(SCRATCH_BUFFER_SIZE, mem.SbType.INT)
    xac_full, xac_full_h = mem.get_sb(SCRATCH_BUFFER_SIZE, mem.SbType.INT)
    xabc_full, xabc_full_h = mem.get_sb(SCRATCH_BUFFER_SIZE, mem.SbType.INT)

    npoints_xab = gpum.interpolate_x(a, b, out=xab_full)
    npoints_xbc = gpum.interpolate_x(b, c, out=xbc_full)
    npoints_xac = gpum.interpolate_x(a, c, out=xac_full)

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

    mem.free_sb(xabc_full_h, mem.SbType.INT)
    mem.free_sb(xac_full_h, mem.SbType.INT)
    mem.free_sb(xbc_full_h, mem.SbType.INT)
    mem.free_sb(xab_full_h, mem.SbType.INT)

    


def get_triangle_points_bary(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    # I'll need to do bounds detection

    aabb_min, aabb_max = gpum.get_aabb(a, b, c)

    if np.array_equal(aabb_min, aabb_max):
        return

    # https://stackoverflow.com/questions/32208359/is-there-a-multi-dimensional-version-of-arange-linspace-in-numpy
    points = gpum.get_grid(aabb_min, aabb_max)

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