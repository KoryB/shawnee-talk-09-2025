from .constants import *
from . import rasterizer, direct, memory as mem

import numpy as np


class Mesh:
    def __init__(self, vertices: np.ndarray, faces: np.ndarray, colors: np.ndarray):
        self.vertices = vertices
        self.faces = faces
        self.colors = colors
        # self.transform = np.eye(4, dtype=FLOAT_DTYPE)
        # self.rotation_matrix = np.eye(4, dtype=FLOAT_DTYPE)
        # self.translation_matrix = np.eye(4, dtype=FLOAT_DTYPE)
        self.position = np.zeros(3, dtype=FLOAT_DTYPE)
        self.rotation = np.zeros(3, dtype=FLOAT_DTYPE)

        self.transformed_vertices = np.zeros_like(vertices)

    def render(self, projection: np.ndarray) -> INTEGER_DTYPE:
        num_tris = 0
        translation = rasterizer.translate(self.position)
        rotation = rasterizer.rotation(self.rotation[0], self.rotation[1], self.rotation[2])
        # rotation = rasterizer.rotation_y(self.rotation[1])
        world_matrix = translation @ rotation
        transform = projection @ world_matrix

        triangles = self.vertices[self.faces]
        triangle_colors = self.colors[self.faces]

        # TODO: Figure out how to broadcast this properly
        for (a, b, c), tcolors in zip(triangles, triangle_colors):
            ap = transform @ a
            bp = transform @ b
            cp = transform @ c

            num_tris += 1
            direct.draw_triangle(
                rasterizer.to_screen(ap), rasterizer.to_screen(bp), 
                rasterizer.to_screen(cp), tcolors)
                
        return num_tris
