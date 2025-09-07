from . import gpu

import argparse

import pygame
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="3D Graphics Demo for Shawnee Talk")

    return parser.parse_args()


def main(args):
    pygame.init()

    screen = pygame.display.set_mode((640, 480))

    gpu_screen = gpu.screen.SCREEN
    
    gpu.direct.draw_rect(gpu_screen, 16, 32, 64, 128, gpu.screen.Color(0x00, 0x00, 0x00))

    # for i in range(10_000):
        # gpu.draw_line(buff.array, np.array([0, i//10], dtype=int), np.array([i//10, 0], dtype=int), gpu.Color(255, 0, 0).as_numpy())
    # gpu.draw_triangle(buff.array, np.array([64, 64]), np.array([360, 128]), np.array([64, 128]), gpu.Color(255, 0, 0).as_numpy())

    buff_surface_raw = gpu_screen.get_surface()
    buff_target = pygame.Surface((640, 480), flags=pygame.SRCALPHA)
    font = pygame.font.SysFont("Georgia", 16)

    clock = pygame.time.Clock()

    while True:
        # Process player inputs.
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                raise SystemExit
            
            if event.type == pygame.MOUSEBUTTONDOWN:
                gpu.clear(gpu_screen, gpu.DEFAULT_COLOR)

        for i in range(1000):
            gpu.direct.draw_triangle(gpu_screen.array, np.array([0,0]), np.array([0, 120]), np.array([200, 120]), gpu.screen.Color(255, 0, 0).array)

        # Do logical updates here.
        # ...

        

        screen.fill("purple")  # Fill the display with a solid color

        pygame.transform.scale2x(buff_surface_raw.convert_alpha(), buff_target)
        screen.blit(buff_target, (0, 0))
        surf = font.render(f"FPS: {clock.get_fps()}", False, (0, 0, 0))
        screen.blit(surf, (0, 0))

        # Render the graphics here.
        # ...

        pygame.display.flip()  # Refresh on-screen display
        clock.tick(60)         # wait until next frame (at 60 FPS)
