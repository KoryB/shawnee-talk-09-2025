import numpy as np
import scipy.signal as spsi
import pygame

# cm
WIDTH = 256
HEIGHT = 256

LAPLACIAN_KERNEL = np.array([
    [0, 1, 0],
    [1, -4, 1],
    [0, 1, 0]
])

HEAT_COEFF = 1.0


# Uses valid, assumes a one-cell border for boundary
def laplacian(arr: np.array) -> np.array:
    return spsi.convolve2d(arr, LAPLACIAN_KERNEL, mode='valid')


def create_surface(arr: np.array) -> pygame.Surface:
    to_pygame = ((arr/arr.max()) * 255).astype(int)
    to_pygame = np.stack([to_pygame, to_pygame, to_pygame], axis=-1)

    return pygame.surfarray.make_surface(to_pygame)


# We don't need to set corners for laplacian in this case
def set_boundary(arr):
    arr[:, 0] = arr[:, 1]
    arr[:, -1] = arr[:, -2]
    arr[0, :] = arr[1, :]
    arr[-1, :] = arr[-2, :]


def step_simulation(u_with_boundary, dt):
    set_boundary(u_with_boundary)
    del2 = laplacian(u_with_boundary)
    return dt * HEAT_COEFF * del2



def main():
    # Setup
    pygame.init()
    np.random.seed(1234)
    screen = pygame.display.set_mode((WIDTH, HEIGHT))

    u_with_boundary = np.zeros((WIDTH+2, HEIGHT+2))
    u = u_with_boundary[1:-1, 1:-1]

    u[0:100, :] = np.linspace(0, 1, HEIGHT)[np.newaxis, :]

    font = pygame.font.SysFont("Georgia", 16)

    clock = pygame.time.Clock()

    while True:
        # Process player inputs.
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                raise SystemExit
            
            if event.type == pygame.MOUSEWHEEL:
                pass
        
        dt_ms = clock.tick(60)         # wait until next frame (at 30 FPS)

        for _ in range(10):
            u += step_simulation(u_with_boundary, 1/10)

        print(u.sum())

        screen.blit(create_surface(u), (0, 0))

        pygame.display.flip()  # Refresh on-screen display


if __name__ == "__main__":
    main()