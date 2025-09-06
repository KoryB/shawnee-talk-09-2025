from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pygame
from numba import njit, jit


@dataclass
class Color:
    r: np.uint8
    g: np.uint8
    b: np.uint8
    a: np.uint8 = 0xFF

    def as_numpy(self) -> np.ndarray:
        return np.array([self.r, self.g, self.b, self.a], dtype=np.uint8)


SURFACE_MODE = "RGBA"
DEFAULT_COLOR = Color(255, 0, 255)
BYTE_WIDTH = 4


class Buffer:
    def __init__(self, array: np.ndarray):
        self.array = array
        self.mode = SURFACE_MODE

    def get_surface(self):
        h, w, _ = self.array.shape

        surf = pygame.image.frombuffer(self.array.data, (w, h), SURFACE_MODE)

        return surf


def get_buffer(width: int, height: int, color: Color = Color(0xFF, 0x00, 0xFF)) -> Buffer:
    raw = np.tile(color.as_numpy(), (height, width, 1))
    buff = Buffer(raw)

    return buff


def clear(buff: Buffer, color: Color):
    buff.array[:, :, :] = color.as_numpy()


def draw_rect(buff: Buffer, x: int, y: int, w: int, h: int, color: Color):
    buff.array[y:(y+h), x:(x+w), :] = color.as_numpy()[np.newaxis, np.newaxis, :]


def get_aabb(*points: np.ndarray):
    min = np.min(points, axis=0)
    max = np.max(points, axis=0)

    return min, max


def get_grid(min: np.ndarray, max: np.ndarray) -> np.ndarray:
    return np.mgrid[min[0]:max[0], min[1]:max[1]].reshape(2,-1).T


def get_line(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    npoints = np.max(np.array([np.abs(b[0] - a[0]) + 1, np.abs(b[1] - a[1]) + 1]))
    x = np.linspace(a[0], b[0], npoints, dtype=np.int32)
    y = np.linspace(a[1], b[1], npoints, dtype=np.int32)

    return x, y


def interpolate_y(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    npoints = b[0] - a[0] + 1

    return np.linspace(a[1], b[1], npoints, dtype=np.int32)


def interpolate_x(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    npoints = b[1] - a[1] + 1

    return np.linspace(a[0], b[0], npoints, dtype=np.int32)



def draw_line(array: np.ndarray, a: np.ndarray, b: np.ndarray, color: np.ndarray):
    x, y = get_line(a, b)

    array[y, x, :] = color


@njit
def triangle_blit(buff: np.ndarray, x_left: np.ndarray, x_right: np.ndarray, y0: np.int32, y1: np.int32, color: np.ndarray):
    for y in range(y0, y1 + 1):
        for x in range(x_left[y - y0], x_right[y - y0] + 1):
            buff[y, x, :] = color


def draw_triangle(buff: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray, color: np.ndarray):
    # Sort point y values
    if b[1] < a[1]:
        a, b = b, a

    if c[1] < a[1]:
        a, c = c, a

    if c[1] < b[1]:
        b, c = c, b

    xab = interpolate_x(a, b)
    xbc = interpolate_x(b, c)
    xac = interpolate_x(a, c)

    xabc = np.concatenate([xab[:-1], xbc])

    midpoint = np.floor_divide(xabc.size, 2, dtype=np.int32)

    if xac[midpoint] < xabc[midpoint]:
        x_left = xac
        x_right = xabc

    else:
        x_left = xabc
        x_right = xac

    triangle_blit(buff, x_left, x_right, a[1], c[1], color)
    


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

