from typing import Optional
from .constants import *
from . import gpu_math as gpum
from . import memory as mem
from .screen import SCREEN, DEFAULT_COLOR

import numpy as np
from numba import njit


def clear():
    SCREEN.color_buffer[:, :, :] = DEFAULT_COLOR
    SCREEN.depth_buffer[:, :] = 1.0


def draw_triangle(
        a: np.ndarray, b: np.ndarray, c: np.ndarray, colors: np.ndarray):
    
    """
    Either all buffers or none should be set
    """

    bary = np.zeros(3, dtype=FLOAT_DTYPE)

    # Sort point y values
    a_orig = np.copy(a)
    b_orig = np.copy(b)
    c_orig = np.copy(c)

    if b[1] < a[1]:
        a, b = b, a

    if c[1] < a[1]:
        a, c = c, a

    if c[1] < b[1]:
        b, c = c, b

    ai = np.round(a).astype(INTEGER_DTYPE)
    bi = np.round(b).astype(INTEGER_DTYPE)
    ci = np.round(c).astype(INTEGER_DTYPE)

    xab = gpum.interpolate_x(ai, bi)
    xbc = gpum.interpolate_x(bi, ci)
    xac = gpum.interpolate_x(ai, ci)

    xabc = np.concatenate([xab[:-1], xbc])

    midpoint = np.floor_divide(xabc.size, 2, dtype=UNSIGNED_INTEGER_DTYPE)

    if xac[midpoint] < xabc[midpoint]:
        x_left = xac
        x_right = xabc

    else:
        x_left = xabc
        x_right = xac

    x_left = np.clip(x_left, 0,  SCREEN_WIDTH - 1)
    x_right = np.clip(x_right, 0, SCREEN_WIDTH - 1)

    y_top = ai[1]
    y_bottom = ci[1]

    if ai[1] < 0:
        y_top = 0
        x_left = x_left[-ai[1]:]
        x_right = x_right[-ai[1]:]

    if ci[1] > SCREEN_HEIGHT - 1:
        y_bottom = SCREEN_HEIGHT - 1
        x_left = x_left[:ci[1]]
        x_right = x_right[:ci[1]]

    triangle_blit(
        x_left, x_right, y_top, y_bottom, 
        a_orig, b_orig, c_orig, 
        bary, colors, 
        SCREEN.color_buffer, SCREEN.depth_buffer)


# Numba stuff
def triangle_blit(
        x_left: np.ndarray, x_right: np.ndarray, y0: INTEGER_DTYPE, y1: INTEGER_DTYPE, 
        a: np.ndarray, b: np.ndarray, c: np.ndarray, 
        bary_temp: np.ndarray, colors: np.ndarray, color_buffer: np.ndarray, depth_buffer: np.ndarray):
    for y in range(y0, y1 + 1):
        i = y - y0

        for x in range(x_left[i], x_right[i] + 1):
            p = np.array([x, y], dtype=INTEGER_DTYPE)

            compute_barycentric_coordinates(p, c[0:2], a[0:2], b[0:2], bary_temp)

            if bary_temp[0] > -1:
                d = bary_temp[0]*a[2] + bary_temp[1]*b[2] + bary_temp[2]*c[2]

                if d < depth_buffer[y, x]:
                    color_buffer[y, x, :] = bary_temp[0]*colors[0] + bary_temp[1]*colors[1] + bary_temp[2]*colors[2]
                    # TODO KB: Mention this bug and solving it
                    # color_buffer[y, x, 0] = bary_temp[0]*255
                    # color_buffer[y, x, 1] = bary_temp[1]*255
                    # color_buffer[y, x, 2] = bary_temp[2]*255
                    # color_buffer[y, x, 3] = 255
                    depth_buffer[y, x] = d
    
    
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