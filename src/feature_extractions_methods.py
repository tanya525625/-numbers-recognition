import numpy as np

import src.utils as utils


def make_mean_value_in_square(image, window=10):
    image = utils.resize_image(image, window)
    squares_mean_values = []
    h, w = image.shape
    for i in range(0, h, window):
        for j in range(0, w, window):
            square = image[i:i + window, j:j + window]
            squares_mean_values.append(np.mean(square))
    return squares_mean_values
