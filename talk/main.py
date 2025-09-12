from . import gpu

import argparse

import pygame
import numpy as np
import trimesh


MESH_FILE = "mesh/kirby.glb"


def parse_args():
    parser = argparse.ArgumentParser(description="3D Graphics Demo for Shawnee Talk")

    return parser.parse_args()


def main(args):
    pygame.init()

    screen = pygame.display.set_mode((640, 480))

    mesh = trimesh.load(MESH_FILE, force='mesh')
    colors = np.array(mesh.visual.vertex_colors, gpu.BYTE_DTYPE)
    vertices = np.array(mesh.vertices, dtype=gpu.FLOAT_DTYPE)
    vertices = np.column_stack([vertices, np.ones(len(vertices), dtype=gpu.FLOAT_DTYPE)])
    faces = np.array(mesh.faces, dtype=gpu.UNSIGNED_INTEGER_DTYPE)

    world_matrix = gpu.rasterizer.translate(np.array([0, -0.15, -0.5], dtype=gpu.FLOAT_DTYPE))

    gpu_screen = gpu.screen.SCREEN
    
    gpu.direct.draw_rect(gpu_screen, 16, 32, 64, 128, gpu.screen.Color(0x00, 0x00, 0x00))

    buff_surface_raw = gpu_screen.get_surface()
    buff_target = pygame.Surface((640, 480), flags=pygame.SRCALPHA)
    font = pygame.font.SysFont("Georgia", 16)

    clock = pygame.time.Clock()
    spin_angle = 0.0

    projection = gpu.rasterizer.projection(90, gpu.SCREEN_ASPECT_RATIO, 0.05)  # What should f be? Something about view angle?
    x = 0
    y = 0
    z = -0.5

    a = np.array([x - 0.3, y, -0.5, 1])
    b = np.array([x + 0.3, y, -0.5, 1])
    c = np.array([x, y + 0.3, z, 1])

    ap = projection @ a
    bp = projection @ b
    cp = projection @ c

    rotation_matrix = np.eye(4, dtype=gpu.FLOAT_DTYPE)

    while True:
        # Process player inputs.
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                raise SystemExit
            
            if event.type == pygame.MOUSEBUTTONDOWN:
                pass

                ap = projection @ a
                bp = projection @ b
                cp = projection @ c

        spin_angle += 5
        gpu.rasterizer.rotation_y(np.deg2rad(spin_angle), out=rotation_matrix)
        triangles = vertices[faces]
        triangle_colors = colors[faces]

        # TODO: Figure out how to broadcast this properly
        for (a, b, c), tcolors in zip(triangles, triangle_colors):
            ap = projection @ world_matrix @ rotation_matrix @ a
            bp = projection @ world_matrix @ rotation_matrix @ b
            cp = projection @ world_matrix @ rotation_matrix @ c

            if gpu.rasterizer.is_in_clip(ap, bp, cp):
                gpu.direct.draw_triangle(
                    gpu_screen.color_buffer, gpu.rasterizer.to_screen(ap), 
                    gpu.rasterizer.to_screen(bp), gpu.rasterizer.to_screen(cp), 
                    tcolors)

        # num_tris = 600
        # xs = np.random.randint(gpu.screen.SCREEN_WIDTH // 2, size=3*num_tris)
        # ys = np.random.randint(gpu.screen.SCREEN_HEIGHT // 2, size=3*num_tris)
        # for i in range(num_tris):
        #     gpu.direct.draw_triangle(
        #         gpu_screen.array, np.array([xs[i], ys[i]]), np.array([xs[i+1], ys[i+1]]), np.array([xs[i+2], ys[i+2]]), 
        #         gpu.screen.Color(255 * i/num_tris, 0, 128 * i/num_tris).array)

        # Do logical updates here.
        # ...

        

        pygame.transform.scale2x(buff_surface_raw.convert_alpha(), buff_target)
        gpu.direct.clear(gpu.screen.SCREEN, gpu.screen.Color(64, 64, 64))
        screen.blit(buff_target, (0, 0))
        surf = font.render(f"FPS: {clock.get_fps()}", False, (0, 0, 0))
        screen.blit(surf, (0, 0))

        pygame.display.flip()  # Refresh on-screen display
        clock.tick(30)         # wait until next frame (at 30 FPS)
