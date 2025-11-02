from .constants import *

from enum import Enum
from typing import Tuple, Union

import numpy as np


class SbType(Enum):
    INT = 0
    UINT = 1
    FLOAT = 2


# Could optimize to work as an object pool type thing
SCRATCH_BUFFER_INT = np.zeros((NUM_SCRATCH_BUFFERS, SCRATCH_BUFFER_SIZE), dtype=INTEGER_DTYPE)
SCRATCH_BUFFER_UINT = np.zeros((NUM_SCRATCH_BUFFERS, SCRATCH_BUFFER_SIZE), dtype=UNSIGNED_INTEGER_DTYPE)
SCRATCH_BUFFER_FLOAT = np.zeros((NUM_SCRATCH_BUFFERS, SCRATCH_BUFFER_SIZE), dtype=FLOAT_DTYPE)

SCRATCH_BUFFER_INT_FREE = np.array([True] * NUM_SCRATCH_BUFFERS, dtype=bool)
SCRATCH_BUFFER_UINT_FREE = np.array([True] * NUM_SCRATCH_BUFFERS, dtype=bool)
SCRATCH_BUFFER_FLOAT_FREE = np.array([True] * NUM_SCRATCH_BUFFERS, dtype=bool)

SCRATCH_BUFFERS = {
    SbType.INT: SCRATCH_BUFFER_INT,
    SbType.UINT: SCRATCH_BUFFER_UINT,
    SbType.FLOAT: SCRATCH_BUFFER_FLOAT,
}

SCRATCH_BUFFER_FREES = {
    SbType.INT: SCRATCH_BUFFER_INT_FREE,
    SbType.UINT: SCRATCH_BUFFER_UINT_FREE,
    SbType.FLOAT: SCRATCH_BUFFER_FLOAT_FREE,
}

SEQUENCE_INT = np.arange(MAX_INT, dtype=INTEGER_DTYPE)
SEQUENCE_UINT = np.arange(MAX_UINT, dtype=UNSIGNED_INTEGER_DTYPE)
BUFFER_MGRID_Y, BUFFER_MGRID_X = np.mgrid[:SCREEN_HEIGHT, :SCREEN_WIDTH].astype(UNSIGNED_INTEGER_DTYPE)

SCREEN_BUFFER = np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH, 4), dtype=BYTE_DTYPE)
DEPTH_BUFFER = np.ones((SCREEN_HEIGHT, SCREEN_WIDTH), dtype=FLOAT_DTYPE)

def get_sb(size: UNSIGNED_INTEGER_DTYPE, type: SbType) -> Tuple[np.ndarray, UNSIGNED_INTEGER_DTYPE]:
    """
    We take some fps hit with this overhead, but the consistency is nice. Keep for now.
    """
    sbs = SCRATCH_BUFFERS[type]
    free = SCRATCH_BUFFER_FREES[type]

    assert np.any(free)

    handle = np.nonzero(free)[0][0]
    free[handle] = False

    return sbs[handle, :size], handle


def free_sb(handle: UNSIGNED_INTEGER_DTYPE, type: SbType):
    assert not SCRATCH_BUFFER_FREES[type][handle]
    SCRATCH_BUFFER_FREES[type][handle] = True
