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

    projection = gpu.rasterizer.projection(90, gpu.SCREEN_ASPECT_RATIO, 0.1)  # What should f be? Something about view angle?
    x = 0
    y = 0
    z = -0.5

    a = np.array([x - 0.3, y, -0.5, 1])
    b = np.array([x + 0.3, y, -0.5, 1])
    c = np.array([x, y + 0.3, z, 1])

    ap = projection @ a
    bp = projection @ b
    cp = projection @ c

    while True:
        # Process player inputs.
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                raise SystemExit
            
            if event.type == pygame.MOUSEBUTTONDOWN:
                pass

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_a:
                    x -= 0.1
                
                if event.key == pygame.K_d:
                    x += 0.1

                if event.key == pygame.K_w:
                    z -= 0.1
                
                if event.key == pygame.K_s:
                    z += 0.1

                
                a = np.array([x - 0.3, y, -0.5, 1])
                b = np.array([x + 0.3, y, -0.5, 1])
                c = np.array([x, y + 0.3, z, 1])

                ap = projection @ a
                bp = projection @ b
                cp = projection @ c

                print(cp)
                print(gpu.rasterizer.to_screen(cp))

        gpu.direct.draw_triangle(
            gpu_screen.array, gpu.rasterizer.to_screen(ap), 
            gpu.rasterizer.to_screen(bp), gpu.rasterizer.to_screen(cp), 
            gpu.screen.Color(255, 0, 128).array)

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
        gpu.direct.clear(gpu.screen.SCREEN, gpu.screen.Color(0, 0, 0))
        screen.blit(buff_target, (0, 0))
        surf = font.render(f"FPS: {clock.get_fps()}", False, (0, 0, 0))
        screen.blit(surf, (0, 0))

        pygame.display.flip()  # Refresh on-screen display
        clock.tick(30)         # wait until next frame (at 30 FPS)
