import numpy as np
from PIL import Image
import os

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

class Layer:
    def __init__(self, input_size, output_size):
        self.weights = np.random.rand(input_size, output_size)
        self.bias = np.zeros((1, output_size))
        self.output = None

class NeuralNetwork:
    def __init__(self, input_size, hidden_sizes, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.layers = []
        prev_size = input_size
        for size in hidden_sizes:
            self.layers.append(Layer(prev_size, size))
            prev_size = size
        self.layers.append(Layer(prev_size, output_size))

    def forward(self, x):
        for layer in self.layers:
            x = sigmoid(np.dot(x, layer.weights) + layer.bias)
            layer.output = x
        return x

    def train(self, x, y, learning_rate, epochs):
        for epoch in range(epochs):
            output = self.forward(x)
            error = y - output
            if epoch % 1000 == 0:  # Вывод ошибки каждую 1000 эпох
                print(f"Epoch {epoch}, Error: {np.mean(np.abs(error))}")
            for i in range(len(self.layers) - 1, -1, -1):
                layer = self.layers[i]
                delta = error * sigmoid_derivative(layer.output)
                if i != 0:
                    layer.weights += learning_rate * np.dot(self.layers[i - 1].output.T, delta)
                else:
                    layer.weights += learning_rate * np.dot(x.T, delta)
                layer.bias += learning_rate * np.sum(delta, axis=0, keepdims=True)
                error = np.dot(delta, layer.weights.T)


if __name__ == "__main__":
    input_size = 49
    hidden_sizes = [10, 10]
    output_size = 3

    nn = NeuralNetwork(input_size, hidden_sizes, output_size)

    image_directory = "images"

    X_train = []
    y_train = []

    for filename in os.listdir(image_directory):
        if filename.endswith(".jpg"):
            image_path = os.path.join(image_directory, filename)
            image = Image.open(image_path).convert('L')
            image = image.resize((7, 7))
            image_data = np.array(image)
            image_data = image_data / 255.0  # Normalizing pixel values

            X_train.append(image_data.flatten())

            label = filename.split("_")[0]
            if label == "C":
                y_train.append([1, 0, 0])
            elif label == "S":
                y_train.append([0, 1, 0])
            elif label == "T":
                y_train.append([0, 0, 1])

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    learning_rate = 0.15
    epochs = 9000
    nn.train(X_train, y_train, learning_rate, epochs)

    test_image = Image.open("test/T_I.jpg").convert('L')
    test_image = test_image.resize((7, 7))
    test_input = np.array(test_image)
    test_input = test_input / 255.0  # Normalizing pixel values
    test_input = test_input.flatten()

    predicted_output = nn.forward(test_input)

    class_names = ["Круг", "Квадрат", "Треугольник"]
    predicted_class = np.argmax(predicted_output)
    print(f"Предсказанный класс: {class_names[predicted_class]}")
    print(predicted_output)
