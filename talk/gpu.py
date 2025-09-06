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
