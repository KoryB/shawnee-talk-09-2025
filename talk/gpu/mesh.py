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

        xab_full, xab_full_h = mem.get_sb(SCRATCH_BUFFER_SIZE, mem.SbType.INT)
        xbc_full, xbc_full_h = mem.get_sb(SCRATCH_BUFFER_SIZE, mem.SbType.INT)
        xac_full, xac_full_h = mem.get_sb(SCRATCH_BUFFER_SIZE, mem.SbType.INT)
        xabc_full, xabc_full_h = mem.get_sb(SCRATCH_BUFFER_SIZE, mem.SbType.INT)
        bary, bary_h = mem.get_sb(3, mem.SbType.FLOAT)

        # TODO: Figure out how to broadcast this properly
        for (a, b, c), tcolors in zip(triangles, triangle_colors):
            ap = transform @ a
            bp = transform @ b
            cp = transform @ c

            if rasterizer.is_in_clip(ap, bp, cp):
                num_tris += 1
                direct.draw_triangle(
                    rasterizer.to_screen(ap), rasterizer.to_screen(bp), 
                    rasterizer.to_screen(cp), tcolors, 
                    xab_full, xbc_full, xac_full,
                    xabc_full, bary)
                
        mem.free_sb(bary_h, mem.SbType.FLOAT)
        mem.free_sb(xabc_full_h, mem.SbType.INT)
        mem.free_sb(xac_full_h, mem.SbType.INT)
        mem.free_sb(xbc_full_h, mem.SbType.INT)
        mem.free_sb(xab_full_h, mem.SbType.INT)
                
        return num_tris
