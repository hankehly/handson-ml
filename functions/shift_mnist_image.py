import numpy as np
from scipy.ndimage.interpolation import shift

DIRECTIONS = {'up', 'down', 'left', 'right'}


def shift_mnist_image(image: np.ndarray, direction: DIRECTIONS, num_px: int = 1) -> np.ndarray:
    assert direction in DIRECTIONS

    shifted_image = image

    if direction == 'up':
        shifted_image = shift(image, [-num_px, 0])
    elif direction == 'down':
        shifted_image = shift(image, [num_px, 0])
    elif direction == 'right':
        shifted_image = shift(image, [0, num_px])
    else:
        shifted_image = shift(image, [0, -num_px])

    return shifted_image
