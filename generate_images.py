import random
import csv
from PIL import Image, ImageDraw
import numpy as np
import cv2 


# Нужно нарисовать закрашенные и незакрашенные фигуры, повернутые на рандомный угол

import os

os.makedirs("dataset", exist_ok=True)
os.makedirs("dataset/rectangle", exist_ok=True)
os.makedirs("dataset/circle", exist_ok=True)
os.makedirs("dataset/triangle", exist_ok=True)


def count_corners(img_pil: Image):
    """
    Определяет количество углов на изображении сгенерированной фигуры.
    
    Параметры:
        img_pil (PIL.Image): Изображение в режиме градаций серого ("L").
        
    Возвращает:
        int: Количество углов.
    """
    # Конвертируем PIL.Image в numpy-массив
    img_np = np.array(img_pil)
    
    # Бинаризация изображения (чёрно-белый формат)
    _, thresh = cv2.threshold(img_np, 127, 255, cv2.THRESH_BINARY)
    
    # Находим контуры
    contours, _ = cv2.findContours(
        thresh, 
        cv2.RETR_EXTERNAL, 
        cv2.CHAIN_APPROX_SIMPLE
    )
    
    if not contours:
        return 0
    
    # Берём самый большой контур (основная фигура)
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Аппроксимируем контур многоугольником
    epsilon = 0.02 * cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)
    
    return len(approx)

def generate_random_shape(width=128, height=128):
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    
    # Случайные параметры
    shape = random.choice(["circle", "rectangle", "triangle"])
    color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    size = random.randint(20, 60)
    x0 = random.randint(10, width - size - 10)
    y0 = random.randint(10, height - size - 10)
    
    is_filled = random.randint(0, 1)

    print(is_filled)

    if (is_filled): fill = color
    else:           fill = "white"



    # Рисование
    if shape == "circle":
        draw.ellipse((x0, y0, x0 + size, y0 + size), fill = fill, outline = color)
    elif shape == "rectangle":
        draw.rectangle((x0, y0, x0 + size, y0 + size), fill= fill, outline = color)
    elif shape == "triangle":
        points = [
            (x0 + size//2, y0),
            (x0, y0 + size),
            (x0 + size, y0 + size)
        ]
        draw.polygon(points, fill = fill, outline = color)


    image = image.rotate(random.randint(0, 360), expand=True, fillcolor="white")

    ImageDraw.floodfill(
        image=image,
        xy=(0, 0),     # Стартовая точка
        value=(255, 255, 255),      # Новый цвет
        thresh=10        # Порог чувствительности (0-255)
    )
    
    return image, shape

# Пример

img = generate_random_shape()

how_many_figures = {"circle" : 0, "rectangle" : 0, "triangle" : 0}



with open("metadata.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["filename", "shape"])
    
    for i in range(100):
        img, shape = generate_random_shape()

        how_many_figures[shape] += 1

        filename = f"shape_{i}.png"
        img.save(f"dataset/{shape}/{filename}")
        writer.writerow([filename, shape])


img.save("random_shape.png")
print(how_many_figures)