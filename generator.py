from PIL import Image, ImageDraw
import random
import os

# Папка для сохранения изображений
output_folder = "generated_images"
os.makedirs(output_folder, exist_ok=True)

# Функция для создания битых изображений
def create_corrupted_image(image, corruption_factor=0.1):
    width, height = image.size
    total_pixels = width * height
    num_corrupted_pixels = int(corruption_factor * total_pixels)

    for _ in range(num_corrupted_pixels):
        x = random.randint(0, width - 1)
        y = random.randint(0, height - 1)
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        image.putpixel((x, y), color)

# Создание и сохранение изображений
for shape in ["circle", "square", "triangle"]:
    for i in range(5):
        image = Image.new("RGB", (7, 7), (255, 255, 255))
        draw = ImageDraw.Draw(image)
        shape_color = (0, 0, 0)

        # Рисование круга внутри квадрата 5x5
        if shape == "circle":
            square_size = 5
            x_offset = random.randint(0, 7 - square_size)
            y_offset = random.randint(0, 7 - square_size)
            square_top_left = (x_offset, y_offset)
            square_bottom_right = (square_top_left[0] + square_size - 1, square_top_left[1] + square_size - 1)
            draw.rectangle([square_top_left, square_bottom_right], outline=shape_color)
            circle_radius = square_size // 2
            circle_center = (square_top_left[0] + circle_radius, square_top_left[1] + circle_radius)
            draw.ellipse(
                (circle_center[0] - circle_radius, circle_center[1] - circle_radius,
                 circle_center[0] + circle_radius, circle_center[1] + circle_radius),
                outline=shape_color)

        # Рисование квадрата 5x5
        elif shape == "square":
            square_size = 5
            x_offset = random.randint(0, 7 - square_size)
            y_offset = random.randint(0, 7 - square_size)
            square_top_left = (x_offset, y_offset)
            square_bottom_right = (square_top_left[0] + square_size - 1, square_top_left[1] + square_size - 1)
            draw.rectangle([square_top_left, square_bottom_right], outline=shape_color)

        # Рисование равнобедренного треугольника с основанием 7
        elif shape == "triangle":
            triangle_base = 7
            triangle_height = triangle_base
            x_offset = random.randint(0, 7 - triangle_base)
            y_offset = random.randint(0, 7 - triangle_height)
            x1 = x_offset
            y1 = y_offset
            x2 = x_offset + triangle_base - 1
            y2 = y_offset
            x3 = x_offset + triangle_base // 2
            y3 = y_offset + triangle_height - 1
            draw.polygon([(x1, y1), (x2, y2), (x3, y3)], outline=shape_color)

        # Создание и сохранение битого изображения
        file_name = f"{shape[0].upper()}-{i + 1}.jpg"
        file_path = os.path.join(output_folder, file_name)
        create_corrupted_image(image)
        image.save(file_path)

print("Генерация изображений завершена.")