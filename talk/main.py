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


        num_tris = 600
        xs = np.random.randint(gpu.screen.SCREEN_WIDTH // 2, size=3*num_tris)
        ys = np.random.randint(gpu.screen.SCREEN_HEIGHT // 2, size=3*num_tris)
        for i in range(num_tris):
            gpu.direct.draw_triangle(
                gpu_screen.array, np.array([xs[i], ys[i]]), np.array([xs[i+1], ys[i+1]]), np.array([xs[i+2], ys[i+2]]), 
                gpu.screen.Color(255 * i/num_tris, 0, 128 * i/num_tris).array)

        # Do logical updates here.
        # ...

        

        pygame.transform.scale2x(buff_surface_raw.convert_alpha(), buff_target)
        gpu.direct.clear(gpu.screen.SCREEN, gpu.screen.Color(0, 0, 0))
        screen.blit(buff_target, (0, 0))
        surf = font.render(f"FPS: {clock.get_fps()}", False, (0, 0, 0))
        screen.blit(surf, (0, 0))

        pygame.display.flip()  # Refresh on-screen display
        clock.tick(30)         # wait until next frame (at 30 FPS)
