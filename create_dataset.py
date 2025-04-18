import os
import sqlite3
from image_func import generate_random_shape, detect_shi_tomasi_corners


def normalise_n_corners(img):
    n_corners = detect_shi_tomasi_corners(img)

    if n_corners <= 3:  return 1
    if n_corners == 4:  return 4
    if n_corners  > 4:  return 9 
    
    
table_name = 'dataset'
db_name    = 'dataset.db'
n_images = 50


# Подключение к БД
conn = sqlite3.connect(f'dataset/{db_name}')
cursor = conn.cursor()


# Создание таблицы
cursor.execute('''
CREATE TABLE IF NOT EXISTS dataset (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    num_verticles INTEGER,
    figure_type TEXT NOT NULL
)
''')

os.makedirs("dataset", exist_ok=True)
'''os.makedirs("dataset/rectangle", exist_ok=True)
os.makedirs("dataset/circle", exist_ok=True)
os.makedirs("dataset/triangle", exist_ok=True)'''


how_many_figures = {"circle" : 0, "rectangle" : 0, "triangle" : 0}


cursor.execute(f"SELECT COUNT(*) FROM {table_name};")
row_count = cursor.fetchone()[0]  #количество существующих строк в таблице для синхронизации с номерами изображений

for i in range(row_count, row_count + n_images):
    img, shape = generate_random_shape()

    how_many_figures[shape] += 1
    
    filename = f"shape_{i + 1}.png"
    img.save(f"dataset/{filename}")


    feature = normalise_n_corners(img)


    cursor.execute("INSERT INTO dataset (num_verticles, figure_type) VALUES (?, ?)", (feature, shape))


print(how_many_figures)


conn.commit()  
cursor.close()
conn.close()