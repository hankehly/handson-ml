import numpy as np
from scipy.ndimage.interpolation import shift

DIRECTIONS = {'up', 'down', 'left', 'right'}


def shift_mnist_image(image: np.ndarray, direction: str, px: int = 1) -> np.ndarray:
    if not direction in DIRECTIONS: raise ValueError('invalid direction')

    if direction == 'up':
        shifted_image = shift(image, [-px, 0])
    elif direction == 'down':
        shifted_image = shift(image, [px, 0])
    elif direction == 'right':
        shifted_image = shift(image, [0, px])
    else:
        shifted_image = shift(image, [0, -px])

    return shifted_image
