import numpy as np
from PIL import Image

def image_to_numpy(image_path, label):
    image = Image.open(image_path)
    image = image.convert('RGB')
    return np.array(image), label


h = '/home/nathan/PycharmProjects/AI_less/Stamp/'
image_paths_and_labels = []
#
for i in range(1, 13):
    image_paths_and_labels.append(f'{h}{i}.jpg')

images_numpy = []
labels = []
for image_path, label in image_paths_and_labels:
    images_numpy.append(image_to_numpy(image_path, label))
    labels.append(label)

np.savez('images.npz', images=images_numpy, labels=labels)
