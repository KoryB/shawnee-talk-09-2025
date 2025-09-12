from .constants import *
from . import memory as mem

import numpy as np
import pygame


"""
Everything is assumed to be single threaded (yay)

And single screen (less yay)
"""


DEFAULT_COLOR = np.array([64, 64, 64, 255], dtype=BYTE_DTYPE)


class Screen:
    def __init__(self, color_buffer, depth_buffer):
        self.color_buffer = color_buffer
        self.depth_buffer = depth_buffer

    
    def get_surface(self):
        h, w, _ = self.color_buffer.shape

        surf = pygame.image.frombuffer(self.color_buffer.data, (w, h), SURFACE_MODE)

        return surf

SCREEN = Screen(mem.SCREEN_BUFFER, mem.DEPTH_BUFFER)

