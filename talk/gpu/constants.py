import numpy as np


BYTE_DTYPE = np.uint8
INTEGER_DTYPE = np.int32
UNSIGNED_INTEGER_DTYPE = np.uint32
FLOAT_DTYPE = np.float32

# For our system these are max, however they cause overflow if the actual dtype is this small
MAX_INT = np.iinfo(np.int16).max
MAX_UINT = np.iinfo(np.uint16).max


# only 32bit, 24bit no alpha, or 8bit pallete
SURFACE_MODE = "RGBA"
SCREEN_WIDTH = 640
SCREEN_HEIGHT = 480
SCREEN_ASPECT_RATIO = SCREEN_WIDTH / SCREEN_HEIGHT
NUM_SCRATCH_BUFFERS = 32
SCRATCH_BUFFER_SIZE = MAX_UINT

SCREEN_HALF_WIDTH = SCREEN_WIDTH // 2
SCREEN_HALF_HEIGHT = SCREEN_HEIGHT // 2

SCREEN_HALF_SIZE = np.array([SCREEN_HALF_WIDTH, SCREEN_HALF_HEIGHT], dtype=INTEGER_DTYPE)