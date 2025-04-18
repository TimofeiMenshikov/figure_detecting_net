import cv2
import numpy as np

import random
from PIL import Image, ImageDraw



def detect_shi_tomasi_corners(
    img_pil: Image, 
    max_corners=100, 
    quality_level=0.1, 
    min_distance=3
):
    """
    Детектирует углы методом Shi-Tomasi.
    
    Параметры:
        max_corners - максимальное количество углов
        quality_level - порог качества (0.01-0.1)
        min_distance - минимальное расстояние между углами
    """
    img_np = np.array(img_pil)
    
    # Конвертация в grayscale
    if len(img_np.shape) == 3:
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_np
    
    # Детекция углов
    corners = cv2.goodFeaturesToTrack(
        gray, 
        maxCorners=max_corners,
        qualityLevel=quality_level,
        minDistance=min_distance
    )
    
    # Возвращение количества углов
    if corners is not None:

        return len(corners)
    
    return 0


def generate_random_shape(width=128, height=128):
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    # Случайные параметры
    shape = random.choice(["circle", "rectangle", "triangle"])
    color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    size = random.randint(70, 100)
    x0 = random.randint(10, width - size - 10)
    y0 = random.randint(10, height - size - 10)

    #is_filled = random.randint(0, 1) #(есть возможность генерировать также незакрашенные фигуры, но пока нейросеть не может справиться с таким датасетом)
    is_filled = 1


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

    return image, shape