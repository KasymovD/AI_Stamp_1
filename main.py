import numpy as np
import matplotlib.pyplot as plt
import utils

images, labels = utils.load_dataset()

print(f"Shape of images: {images.shape}")
print(f"Shape of labels: {labels.shape}")

input_size = images.shape[1]
hidden_layer_size = 20
output_size = 1

weights_input_to_hidden = np.random.uniform(-0.5, 0.5, (hidden_layer_size, input_size))
weights_hidden_to_output = np.random.uniform(-0.5, 0.5, (output_size, hidden_layer_size))
bias_input_to_hidden = np.zeros((hidden_layer_size, 1))
bias_hidden_to_output = np.zeros((output_size, 1))

epochs = 200
learning_rate = 0.01

for epoch in range(epochs):
    e_loss = 0
    e_correct = 0

    print(f"Epoch №{epoch + 1}")

    for image, label in zip(images, labels):
        image = np.reshape(image, (-1, 1))
        label = np.reshape(label, (-1, 1))

        hidden_raw = bias_input_to_hidden + weights_input_to_hidden @ image
        hidden = 1 / (1 + np.exp(-hidden_raw))  # sigmoid

        output_raw = bias_hidden_to_output + weights_hidden_to_output @ hidden
        output = 1 / (1 + np.exp(-output_raw))

        e_loss += 1 / len(output) * np.sum((output - label) ** 2, axis=None)
        e_correct += int((output > 0.5) == label)

        delta_output = output - label
        weights_hidden_to_output += -learning_rate * delta_output @ hidden.T
        bias_hidden_to_output += -learning_rate * delta_output

        delta_hidden = (weights_hidden_to_output.T @ delta_output) * (hidden * (1 - hidden))
        weights_input_to_hidden += -learning_rate * delta_hidden @ image.T
        bias_input_to_hidden += -learning_rate * delta_hidden

    print(f"Loss: {round((e_loss / images.shape[0]) * 100, 3)}%")
    print(f"Accuracy: {round((e_correct / images.shape[0]) * 100, 3)}%")

test_image_path = "67.jpg"
test_image = plt.imread(test_image_path, format="jpeg")


def gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])


test_image = 1 - (gray(test_image).astype("float32") / 255)

test_image = np.reshape(test_image, (test_image.shape[0] * test_image.shape[1], 1))

image = np.reshape(test_image, (-1, 1))

hidden_raw = bias_input_to_hidden + weights_input_to_hidden @ image
hidden = 1 / (1 + np.exp(-hidden_raw))  # sigmoid
output_raw = bias_hidden_to_output + weights_hidden_to_output @ hidden
output = 1 / (1 + np.exp(-output_raw))

plt.imshow(test_image.reshape(327, 325), cmap="Greys")
plt.title(f"NN suggests the CUSTOM stamp is: {'real' if output.item() > 0.9 else 'fake'}")
plt.show()
