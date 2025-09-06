from . import gpu

import argparse

import pygame


def parse_args():
    parser = argparse.ArgumentParser(description="3D Graphics Demo for Shawnee Talk")

    return parser.parse_args()


def main(args):
    pygame.init()

    screen = pygame.display.set_mode((1280,720))

    buff = gpu.get_buffer(720, 720)
    buff[0:100, 0:100] = 0xFF000000 # TODO KB: endianness
    square = pygame.image.frombuffer(buff.data, buff.shape, gpu.SURFACE_MODE)

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
