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
    # Setup
    pygame.init()
    np.random.seed(1234)
    screen = pygame.display.set_mode((640, 480))

    mesh = trimesh.load(MESH_FILE, force='mesh')
    colors = np.array(mesh.visual.vertex_colors, gpu.BYTE_DTYPE)
    vertices = np.array(mesh.vertices, dtype=gpu.FLOAT_DTYPE)
    vertices = np.column_stack([vertices, np.ones(len(vertices), dtype=gpu.FLOAT_DTYPE)])
    faces = np.array(mesh.faces, dtype=gpu.UNSIGNED_INTEGER_DTYPE)

    buff_surface_raw = gpu.screen.SCREEN.get_surface()
    # buff_target = pygame.Surface((640*2, 480*2), flags=pygame.SRCALPHA)
    font = pygame.font.SysFont("Georgia", 16)

    clock = pygame.time.Clock()

    # variables
    kirbies = []

    for r in range(3):
        for c in range(3):
            if r == 1 or c == 1:
                x = [-0.7, 0.0, 0.7][c]
                y = [-0.7, 0.0, 0.7][r]

                kirby = gpu.Mesh(vertices, faces, colors)
                kirby.movement_angle = 0.0
                kirby.rotation_speed = r + c + 2
                kirby.rotation_angle = (1 + r ^ c) % 2

                kirby.original_position = np.array([x, y, -0.8 - np.abs(x*0.5) - np.abs(y*0.5)], dtype=gpu.FLOAT_DTYPE)
                kirby.position = np.copy(kirby.original_position)

                kirbies.append(kirby)

    projection = gpu.rasterizer.projection(90, gpu.SCREEN_ASPECT_RATIO, 0.1)  # What should f be? Something about view angle?

    while True:
        # Process player inputs.
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                raise SystemExit
            
            if event.type == pygame.MOUSEWHEEL:
                pass

        # Update game
        for kirby in kirbies:
            kirby.movement_angle += np.deg2rad(kirby.rotation_speed)
            kirby.position = kirby.original_position + np.array([
                0.1*np.cos(kirby.movement_angle),
                0.1*np.sin(kirby.movement_angle),
                0.1*np.sin(kirby.movement_angle)*np.cos(kirby.movement_angle),
            ])
            kirby.rotation[kirby.rotation_angle] += np.deg2rad(kirby.rotation_speed)
            kirby.rotation[(kirby.rotation_angle+1) % 3] += np.deg2rad(kirby.rotation_speed)*2

        tri_count = 0
        # Render game objects
        for kirby in kirbies:
            tri_count += kirby.render(projection)

        # Render to the screen
        screen.blit(buff_surface_raw, (0, 0))
        gpu.direct.clear()
        surf = font.render(f"FPS: {clock.get_fps()}", False, (255, 255, 255))
        surf_tri_count = font.render(f"Triangle Count: {tri_count}", False, (255, 255, 255))
        screen.blit(surf, (0, 0))
        screen.blit(surf_tri_count, (0, 24))

        pygame.display.flip()  # Refresh on-screen display
        clock.tick()         # wait until next frame (at 30 FPS)
