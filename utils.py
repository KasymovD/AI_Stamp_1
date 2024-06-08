import numpy as np


def load_dataset():
    num_samples = 100
    image_height = 327
    image_width = 325

    images = np.random.rand(num_samples, image_height * image_width)
    labels = np.random.randint(0, 2, (num_samples, 1))

    return images, labels
