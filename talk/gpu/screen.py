from .constants import *
from . import memory as mem

import numpy as np
import pygame


"""
Everything is assumed to be single threaded (yay)

And single screen (less yay)
"""

class Color:
    def __init__(self, r: np.uint8, g: np.uint8, b: np.uint8, a: np.uint8 = 0xFF):
        self.r = r
        self.g = g
        self.b = b
        self.a = a

        self.array = np.array([r, g, b, a], dtype=np.uint8)


DEFAULT_COLOR = Color(255, 0, 255)


class Screen:
    def __init__(self, color_buffer, depth_buffer):
        self.color_buffer = color_buffer
        self.depth_buffer = depth_buffer

    
    def get_surface(self):
        h, w, _ = self.color_buffer.shape

        surf = pygame.image.frombuffer(self.color_buffer.data, (w, h), SURFACE_MODE)

        return surf

SCREEN = Screen(mem.SCREEN_BUFFER, mem.DEPTH_BUFFER)

