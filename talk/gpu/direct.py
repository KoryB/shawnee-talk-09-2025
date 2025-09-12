from .constants import *
from . import gpu_math as gpum
from . import memory as mem
from .screen import Screen, Color, SCREEN

import numpy as np
from numba import njit


def clear(buff: Screen, color: Color):
    buff.color_buffer[:, :, :] = color.array
    buff.depth_buffer[:, :] = 1.0


def draw_rect(buff: Screen, x: int, y: int, w: int, h: int, color: Color):
    buff.color_buffer[y:(y+h), x:(x+w), :] = color.array


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
def triangle_blit(
        x_left: np.ndarray, x_right: np.ndarray, y0: INTEGER_DTYPE, y1: INTEGER_DTYPE, 
        a: np.ndarray, b: np.ndarray, c: np.ndarray, 
        ai: np.ndarray, bi: np.ndarray, ci: np.ndarray,
        bary_temp: np.ndarray, colors: np.ndarray, color_buffer: np.ndarray, depth_buffer: np.ndarray):
    for y in range(y0, y1 + 1):
        i = y - y0

        for x in range(x_left[i], x_right[i] + 1):
            p = np.array([x, y], dtype=INTEGER_DTYPE)

            compute_barycentric_coordinates(p, a[0:2], b[0:2], c[0:2], bary_temp)

            if bary_temp[0] >= 0.0:
                d = bary_temp[0]*a[2] + bary_temp[1]*b[2] + bary_temp[2]*c[2]

                if d < depth_buffer[y, x]:
                    color_buffer[y, x, :] = bary_temp[0]*colors[0] + bary_temp[1]*colors[1] + bary_temp[2]*colors[2]
                    depth_buffer[y, x] = d


def draw_triangle(buff: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray, colors: np.ndarray):
    # Sort point y values
    if b[1] < a[1]:
        a, b = b, a

    if c[1] < a[1]:
        a, c = c, a

    if c[1] < b[1]:
        b, c = c, b

    ai = a.astype(INTEGER_DTYPE)
    bi = b.astype(INTEGER_DTYPE)
    ci = c.astype(INTEGER_DTYPE)

    xab_full, xab_full_h = mem.get_sb(SCRATCH_BUFFER_SIZE, mem.SbType.INT)
    xbc_full, xbc_full_h = mem.get_sb(SCRATCH_BUFFER_SIZE, mem.SbType.INT)
    xac_full, xac_full_h = mem.get_sb(SCRATCH_BUFFER_SIZE, mem.SbType.INT)
    xabc_full, xabc_full_h = mem.get_sb(SCRATCH_BUFFER_SIZE, mem.SbType.INT)
    bary, bary_h = mem.get_sb(3, mem.SbType.FLOAT)

    npoints_xab = gpum.interpolate_x(ai, bi, out=xab_full)
    npoints_xbc = gpum.interpolate_x(bi, ci, out=xbc_full)
    npoints_xac = gpum.interpolate_x(ai, ci, out=xac_full)

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

    x_left = np.clip(x_left, 0, SCREEN_WIDTH)
    x_right = np.clip(x_right, 0, SCREEN_WIDTH)

    y_top = ai[1]
    y_bottom = ci[1]

    if ai[1] < 0:
        y_top = 0
        x_left = x_left[-ai[1]:]
        x_right = x_right[-ai[1]:]

    if ci[1] > SCREEN_HEIGHT:
        y_bottom = SCREEN_HEIGHT
        x_left = x_left[:ci[1]]
        x_right = x_right[:ci[1]]

    # y_top = np.clip(a[1], 0, SCREEN_HEIGHT)
    # y_bottom = np.clip(c[1], 0, SCREEN_HEIGHT)

    """
    TODO KB:
        This ended up being significantly slower, numpy mask blit operations are slow apparently
    """
    # ys = slice(a[1], c[1]+1)
    # x_grid = (x_left[:, np.newaxis] <= BUFFER_MGRID_X[ys, 0:200]) & (BUFFER_MGRID_X[ys, 0:200] <= x_right[:, np.newaxis])
    # buff[ys, 0:200][x_grid] = color

    triangle_blit(
        x_left, x_right, y_top, y_bottom, 
        a, b, c, 
        ai, bi, ci, 
        bary, colors, 
        SCREEN.color_buffer, SCREEN.depth_buffer)

    mem.free_sb(bary_h, mem.SbType.FLOAT)
    mem.free_sb(xabc_full_h, mem.SbType.INT)
    mem.free_sb(xac_full_h, mem.SbType.INT)
    mem.free_sb(xbc_full_h, mem.SbType.INT)
    mem.free_sb(xab_full_h, mem.SbType.INT)

    
@njit
def compute_barycentric_coordinates(p: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray, out=np.ndarray) -> np.ndarray:
    v0 = b - a
    v1 = c - a
    v2_points = p - a

    d00 = np.dot(v0, v0)
    d01 = np.dot(v0, v1)
    d11 = np.dot(v1, v1)
    d20 = np.dot(v2_points, v0)
    d21 = np.dot(v2_points, v1)
    
    denom = d00 * d11 - d01 * d01;

    if denom == 0:
        out[:] = -1

    else:
        out[0] = (d11 * d20 - d01 * d21) / denom
        out[1] = (d00 * d21 - d01 * d20) / denom
        out[2] = 1.0 - out[0] - out[1]


# def get_triangle_points_bary(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
#     # I'll need to do bounds detection

#     aabb_min, aabb_max = gpum.get_aabb(a, b, c)

#     if np.array_equal(aabb_min, aabb_max):
#         return

#     # https://stackoverflow.com/questions/32208359/is-there-a-multi-dimensional-version-of-arange-linspace-in-numpy
#     points = gpum.get_grid(aabb_min, aabb_max)

#     v0 = b - a
#     v1 = c - a
#     v2_points = points - a

#     d00 = np.dot(v0, v0)
#     d01 = np.dot(v0, v1)
#     d11 = np.dot(v1, v1)
#     d20 = np.dot(v2_points, v0)
#     d21 = np.dot(v2_points, v1)
    
#     denom = d00 * d11 - d01 * d01;

#     v = (d11 * d20 - d01 * d21) / denom;
#     w = (d00 * d21 - d01 * d20) / denom;
#     u = 1.0 - v - w;

#     mask = (v > 0) & (w > 0) & (u > 0) & (v + w + u <= 1.0)

#     return points[mask]