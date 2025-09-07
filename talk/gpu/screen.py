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
    def __init__(self, array):
        self.array = array

    
    def get_surface(self):
        h, w, _ = self.array.shape

        surf = pygame.image.frombuffer(self.array.data, (w, h), SURFACE_MODE)

        return surf

SCREEN = Screen(mem.SCREEN_BUFFER)

