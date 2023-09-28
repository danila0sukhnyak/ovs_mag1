import numpy as np
from PIL import Image
import os


# Функция активации - сигмоида
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Производная сигмоиды
def sigmoid_derivative(x):
    return x * (1 - x)


# Класс для создания нейронных слоев
class Layer:
    def __init__(self, input_size, output_size):
        self.weights = np.random.rand(input_size, output_size)
        self.bias = np.zeros((1, output_size))
        self.output = None


# Класс для создания нейронной сети
class NeuralNetwork:
    def __init__(self, input_size, hidden_sizes, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.layers = []

        # Создаем скрытые слои
        prev_size = input_size
        for size in hidden_sizes:
            self.layers.append(Layer(prev_size, size))
            prev_size = size

        # Создаем выходной слой
        self.layers.append(Layer(prev_size, output_size))

    def forward(self, x):
        # Прямое распространение
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

            gradients = []
            for i in range(len(self.layers) - 1, -1, -1):
                layer = self.layers[i]
                delta = error * sigmoid_derivative(layer.output)
                if i != 0:
                    layer_gradient_weights = np.dot(self.layers[i - 1].output.T, delta)
                else:
                    layer_gradient_weights = np.dot(x.T, delta)
                layer_gradient_bias = np.sum(delta, axis=0, keepdims=True)

                gradients.insert(0, (layer_gradient_weights, layer_gradient_bias))
                error = np.dot(delta, layer.weights.T)

            # Обновление весов и смещений
            for i in range(len(self.layers)):
                layer = self.layers[i]
                layer.weights += learning_rate * gradients[i][0]
                layer.bias += learning_rate * gradients[i][1]

    # def train(self, x, y, learning_rate, epochs):
    #     for epoch in range(epochs):
    #         output = self.forward(x)
    #         error = y - output
    #         if epoch % 1000 == 0:  # Вывод ошибки каждую 1000 эпох
    #             print(f"Epoch {epoch}, Error: {np.mean(np.abs(error))}")
    #         for i in range(len(self.layers) - 1, -1, -1):
    #             layer = self.layers[i]
    #             delta = error * sigmoid_derivative(layer.output)
    #             if i != 0:
    #                 layer.weights += learning_rate * np.dot(self.layers[i - 1].output.T, delta)
    #             else:
    #                 layer.weights += learning_rate * np.dot(x.T, delta)
    #             layer.bias += learning_rate * np.sum(delta, axis=0, keepdims=True)
    #             error = np.dot(delta, layer.weights.T)

    # def train(self, x, y, learning_rate, epochs):
    #     for _ in range(epochs):
    #         # Прямое распространение
    #         output = self.forward(x)
    #
    #         # Вычисление ошибки
    #         error = y - output
    #
    #         # Обратное распространение ошибки
    #         for i in range(len(self.layers) - 1, -1, -1):
    #             layer = self.layers[i]
    #             delta = error * sigmoid_derivative(layer.output)
    #             if i != 0:
    #                 layer.weights += learning_rate * np.dot(self.layers[i - 1].output.T, delta)
    #             else:
    #                 layer.weights += learning_rate * np.dot(x.T, delta)
    #             layer.bias += learning_rate * np.sum(delta, axis=0, keepdims=True)
    #             error = np.dot(delta, layer.weights.T)


if __name__ == "__main__":
    # Пример обучения нейронной сети
    input_size = 49  # 7x7 пикселей
    hidden_sizes = [16]  # Количество скрытых слоев и их размеры
    output_size = 3  # Количество классов (круг, квадрат, треугольник)

    # Создаем нейронную сеть
    nn = NeuralNetwork(input_size, hidden_sizes, output_size)

    # Каталог с изображениями
    image_directory = "images"

    X_train = []
    y_train = []

    # Загрузка изображений и их меток
    for filename in os.listdir(image_directory):
        if filename.endswith(".jpg"):
            image_path = os.path.join(image_directory, filename)
            image = Image.open(image_path).convert('L')  # Преобразование в черно-белое изображение
            image = image.resize((7, 7))
            image_data = np.array(image)
            image_data = (image_data < 128).astype(int)
            X_train.append(image_data.flatten())
            output_array = image_data.flatten().reshape(7, 7)
            print(image_path)
            print(output_array)

            # Определение меток по названию файла (предполагается, что файлы названы как C_1.jpg, S_9.jpg и т. д.)
            label = filename.split("_")[0]
            if label == "C":
                y_train.append([1, 0, 0])  # Круг
            elif label == "S":
                y_train.append([0, 1, 0])  # Квадрат
            elif label == "T":
                y_train.append([0, 0, 1])  # Треугольник

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    # Обучение нейронной сети
    learning_rate = 0.25
    epochs = 15000
    nn.train(X_train, y_train, learning_rate, epochs)

    # Пример использования нейронной сети для классификации
    test_image = Image.open("test/T_I.jpg").convert('L')
    test_image = test_image.resize((7, 7))
    test_input = np.array(test_image)
    test_input = (test_input < 128).astype(int)  # Преобразование в бинарный массив (черный=1, белый=0)
    test_input = test_input.flatten()

    output_array = test_input.flatten().reshape(7, 7)
    print(output_array)

    predicted_output = nn.forward(test_input)

    # Вывод результата
    class_names = ["Круг", "Квадрат", "Треугольник"]
    predicted_class = np.argmax(predicted_output)
    print(f"Предсказанный класс: {class_names[predicted_class]}")
    print(predicted_output)
