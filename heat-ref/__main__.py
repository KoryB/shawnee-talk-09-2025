from functools import partial

import numpy as np
import scipy.signal as spsi
import pygame

# cm
WIDTH = 128
HEIGHT = 128

WINDOW_WIDTH = 768
WINDOW_HEIGHT = 768

LAPLACIAN_KERNEL = np.array([
    [0, 1, 0],
    [1, -4, 1],
    [0, 1, 0]
])

HEAT_COEFF = 1.0

COLOR_ZERO = pygame.Color(17, 26, 81)
COLOR_ONE = pygame.Color(240, 50, 50)


def linear(t: float) -> float:
    return t


def step(t_step: float, t: float) -> float:
    return np.where(t < t_step, 0.0, 1.0)


def smoothstep(t: float) -> float:
    return t*t*(3.0 - 2.0*t)


def gradient(a: pygame.Color, b: pygame.Color, t: float, f = None):
    if f is None:
        f = linear

    t = np.clip(t, 0.0, 1.0)

    return np.stack([
        a.r * (1 - f(t)) + b.r * f(t),
        a.g * (1 - f(t)) + b.g * f(t),
        a.b * (1 - f(t)) + b.b * f(t)
    ], axis=-1)


# Uses valid, assumes a one-cell border for boundary
def laplacian(arr: np.array) -> np.array:
    return spsi.convolve2d(arr, LAPLACIAN_KERNEL, mode='valid')


def create_surface(arr: np.array, f = None) -> pygame.Surface:
    to_pygame = gradient(COLOR_ZERO, COLOR_ONE, arr.clip(0.0, 1.0), f=f)

    return pygame.surfarray.make_surface(to_pygame)


# We don't need to set corners for laplacian in this case
def set_boundary_border_equal(arr):
    arr[:, 0] = arr[:, 1]
    arr[:, -1] = arr[:, -2]
    arr[0, :] = arr[1, :]
    arr[-1, :] = arr[-2, :]


def set_boundary_border_zero(arr):
    arr[:, 0] = 0
    arr[:, -1] = 0
    arr[0, :] = 0
    arr[-1, :] = 0


def set_boundary_border_mobius(arr):
    arr[:, 0] = arr[:, -2]
    arr[:, -1] = arr[:, 1]
    arr[0, :] = arr[-2, :]
    arr[-1, :] = arr[1, :]


def step_simulation(u_with_boundary, dt, set_boundary):
    set_boundary(u_with_boundary)
    del2 = laplacian(u_with_boundary)
    return dt * HEAT_COEFF * del2


def main():
    # Setup
    pygame.init()
    np.random.seed(1234)
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))

    u_with_boundary = np.zeros((WIDTH+2, HEIGHT+2))
    u = u_with_boundary[1:-1, 1:-1]

    # u[0:100, :] = np.linspace(0, 1, HEIGHT)[np.newaxis, :]

    px, py = 0,0
    p_power = 1.0
    grad_f_i = 0

    gradients = [linear, partial(step, 0.75), smoothstep]

    font = pygame.font.SysFont("Georgia", 16)

    clock = pygame.time.Clock()

    while True:
        # Process player inputs.
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                raise SystemExit
            
            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == pygame.BUTTON_LEFT:
                    p_power *= -1

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    p_power = 0.0

                elif event.key == pygame.K_f:
                    grad_f_i = (grad_f_i + 1) % len(gradients)

                elif event.key == pygame.K_c:
                    u[:, :] = 0


            if event.type == pygame.MOUSEWHEEL:
                p_power += event.precise_y
        
        dt_ms = clock.tick(60)

        px, py = pygame.mouse.get_pos()
        px = np.clip(int(px / WINDOW_WIDTH * WIDTH), 0, WIDTH-2)
        py = np.clip(int(py / WINDOW_HEIGHT * HEIGHT), 0, HEIGHT-2)
        u[px:px+2, py:py+2] = p_power

        for _ in range(10):
            u += step_simulation(u_with_boundary, 1/10, set_boundary=set_boundary_border_mobius)

        surf = create_surface(u, gradients[grad_f_i])
        surf = pygame.transform.scale(surf, (WINDOW_WIDTH, WINDOW_HEIGHT))

        screen.blit(surf, (0, 0))

        pygame.display.flip()  # Refresh on-screen display


if __name__ == "__main__":
    main()