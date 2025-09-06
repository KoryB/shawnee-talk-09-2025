from . import gpu

import argparse

import pygame
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="3D Graphics Demo for Shawnee Talk")

    return parser.parse_args()


def main(args):
    pygame.init()

    screen = pygame.display.set_mode((1280,720))

    buff = gpu.get_buffer(540, 720)
    
    gpu.draw_rect(buff, 16, 32, 64, 128, gpu.Color(0x00, 0x00, 0x00))
    gpu.draw_triangle(buff, np.array([64, 64]), np.array([128, 128]), np.array([64, 128]), gpu.Color(255, 0, 0))

    square = buff.get_surface()

    clock = pygame.time.Clock()

    while True:
        # Process player inputs.
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                raise SystemExit

        # Do logical updates here.
        # ...

        screen.fill("purple")  # Fill the display with a solid color

        screen.blit(square, ((1280-720)/2, 0))

        # Render the graphics here.
        # ...

        pygame.display.flip()  # Refresh on-screen display
        clock.tick(60)         # wait until next frame (at 60 FPS)
