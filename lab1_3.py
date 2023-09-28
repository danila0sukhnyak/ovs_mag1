from PIL import Image
import numpy as np

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
        self.output = None  # Добавляем атрибут output

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
            layer.output = x  # Устанавливаем значение output для каждого слоя
        return x

    def train(self, x, y, learning_rate, epochs):
        for _ in range(epochs):
            # Прямое распространение
            output = self.forward(x)

            # Вычисление ошибки
            error = y - output

            # Обратное распространение ошибки
            for i in range(len(self.layers) - 1, -1, -1):
                layer = self.layers[i]
                delta = error * sigmoid_derivative(layer.output)
                if i != 0:
                    layer.weights += learning_rate * np.dot(self.layers[i-1].output.T, delta)
                else:
                    layer.weights += learning_rate * np.dot(x.T, delta)
                layer.bias += learning_rate * np.sum(delta, axis=0, keepdims=True)
                error = np.dot(delta, layer.weights.T)

if __name__ == "__main__":
    # Пример обучения нейронной сети
    input_size = 49  # 7x7 пикселей
    hidden_sizes = [18, 12, 6]  # Количество скрытых слоев и их размеры
    output_size = 3  # Количество классов (круг, квадрат, треугольник)

    # Создаем нейронную сеть
    nn = NeuralNetwork(input_size, hidden_sizes, output_size)

    # Примеры входных данных и соответствующих меток (классов)
    # Здесь представлен код для загрузки jpg изображений размером 7x7 пикселей
    image_paths = ["C_1.jpg", "S_9.jpg", "T_1.jpg", "C_2.jpg", "C_3.jpg", "C_4.jpg", "S_1.jpg", "S_2.jpg", "S_3.jpg",
                   "S_4.jpg", "S_5.jpg", "S_6.jpg", "S_7.jpg", "S_8.jpg", "S_10.jpg", "S_11.jpg",  "T_2.jpg", "T_3.jpg",
                   "T_4.jpg", "T_5.jpg", "T_6.jpg", "T_7.jpg", "T_8.jpg"]
    X_train = []

    for path in image_paths:
        image = Image.open(path).convert('L')  # Преобразование в черно-белое изображение
        image = image.resize((7, 7))  # Изменение размера до 7x7 пикселей
        image_data = np.array(image)
        image_data = image_data / 255.0  # Нормализация значений пикселей в диапазоне [0, 1]
        X_train.append(image_data.flatten())
        print(image_data.flatten())

    X_train = np.array(X_train)

    # Метки классов для каждого примера
    y_train = np.array([[1, 0, 0],  # Круг
                        [0, 1, 0],  # Квадрат
                        [0, 0, 1],  # Треугольник
                        [1, 0, 0],
                        [1, 0, 0],
                        [1, 0, 0],
                        [0, 1, 0],
                        [0, 1, 0],
                        [0, 1, 0],
                        [0, 1, 0],
                        [0, 1, 0],
                        [0, 1, 0],
                        [0, 1, 0],
                        [0, 1, 0],
                        [0, 1, 0],
                        [0, 1, 0],
                        [0, 0, 1],
                        [0, 0, 1],
                        [0, 0, 1],
                        [0, 0, 1],
                        [0, 0, 1],
                        [0, 0, 1],
                        [0, 0, 1]])

    # Обучение нейронной сети
    learning_rate = 0.3
    epochs = 100000
    nn.train(X_train, y_train, learning_rate, epochs)

    # Пример использования нейронной сети для классификации
    test_image = Image.open("T_1.jpg").convert('L')
    test_image = test_image.resize((7, 7))
    test_input = np.array(test_image)
    test_input = test_input / 255.0
    test_input = test_input.flatten()

    predicted_output = nn.forward(test_input)

    # Вывод результата
    class_names = ["Круг", "Квадрат", "Треугольник"]
    predicted_class = np.argmax(predicted_output)
    print(f"Предсказанный класс: {class_names[predicted_class]}")
    print(predicted_output)