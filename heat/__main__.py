from functools import partial

import numpy as np
import scipy.signal as spsi
import pygame


WINDOW_WIDTH = 768
WINDOW_HEIGHT = 768

SIMULATION_WIDTH = 128
SIMULATION_HEIGHT = 128


def create_heat_surface(u: np.ndarray) -> pygame.Surface:
    uu = (u.clip(0.0, 1.0) * 255).astype(int)

    to_pygame = np.stack([uu, uu, uu], axis=-1)

    return pygame.surfarray.make_surface(to_pygame)


def render_heat(screen: pygame.Surface, u: np.ndarray):
    surf = create_heat_surface(u)
    surf = pygame.transform.scale(surf, (WINDOW_WIDTH, WINDOW_HEIGHT))
    screen.blit(surf, (0, 0))


pygame.init()
np.random.seed(1234)
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))

u = np.zeros((SIMULATION_WIDTH, SIMULATION_HEIGHT))
center_x = slice(SIMULATION_WIDTH//2 - 3, SIMULATION_WIDTH//2 + 4)
center_y = slice(SIMULATION_HEIGHT//2 - 3, SIMULATION_HEIGHT//2 + 4)
u[center_x, center_y] = 1.0

simulation_time = 0.0

font = pygame.font.SysFont("Georgia", 16)

clock = pygame.time.Clock()

while True:
    # Process player inputs.
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            raise SystemExit
    
    dt_ms = clock.tick(60)
    dt_s = dt_ms / 1000.0

    simulation_time += dt_s
    # simulate heat?

    render_heat(screen, u)
    text = font.render(f"Simulation Time: {simulation_time:.2f}s", True, pygame.Color('aliceblue'))
    screen.blit(text, (0,0))

    pygame.display.flip()  # Refresh on-screen display