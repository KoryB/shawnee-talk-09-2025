from dataclasses import dataclass

import numpy as np
import pygame


SURFACE_MODE = "RGBA"
DEFAULT_COLOR = np.uint32(0xFF00FFFF)
BYTE_WIDTH = 4

@dataclass
class Color:
    r: np.uint8
    g: np.uint8
    b: np.uint8
    a: np.uint8 = 0xFF

    def as_numpy(self) -> np.ndarray:
        return np.array([self.r, self.g, self.b, self.a], dtype=np.uint8)


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
    buff.array[:, :, :] = color.as_numpy()[:, np.newaxis, np.newaxis]


def draw_rect(buff: Buffer, x: int, y: int, w: int, h: int, color: Color):
    buff.array[y:(y+h), x:(x+w), :] = color.as_numpy()[np.newaxis, np.newaxis, :]


def get_aabb(*points: np.ndarray):
    min = np.min(points, axis=0)
    max = np.max(points, axis=0)

    return min, max


# https://gamedev.stackexchange.com/questions/23743/whats-the-most-efficient-way-to-find-barycentric-coordinates
# Currently only accepts ints, operates directly on the buffer
def draw_triangle(buff: Buffer, a: np.ndarray, b: np.ndarray, c: np.ndarray, color: Color):
    # I'll need to do bounds detection

    aabb_min, aabb_max = get_aabb(a, b, c)

    if np.array_equal(aabb_min, aabb_max):
        return

    # https://stackoverflow.com/questions/32208359/is-there-a-multi-dimensional-version-of-arange-linspace-in-numpy
    points = np.mgrid[aabb_min[0]:aabb_max[0], aabb_min[1]:aabb_max[1]].reshape(2,-1).T

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

    triangle_points = points[mask]

    buff.array[triangle_points[:, 1], triangle_points[:, 0], :] = color.as_numpy()[np.newaxis, np.newaxis, :]

