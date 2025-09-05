import numpy as np


SURFACE_MODE = "RGBA"
DEFAULT_COLOR = np.uint32(0xFF00FFFF)


def get_buffer(width: int, height: int, fill_value: np.uint32 = DEFAULT_COLOR) -> np.ndarray:
    return np.full(shape=(width, height), fill_value=fill_value, dtype=np.uint32)
