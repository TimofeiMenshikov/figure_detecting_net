import os
import sqlite3

conn = sqlite3.connect('dataset/dataset.db')
cursor = conn.cursor()
folder_to_remove = 'dataset'


# Удаление всех сгенерированных png файлов
for filename in os.listdir(folder_to_remove):
    # Формируем полный путь
    file_path = os.path.join(folder_to_remove, filename)
    
    # Проверяем, что это файл и имеет расширение .png (без учета регистра)
    if os.path.isfile(file_path) and filename.lower().endswith('.png'):
        try:
            os.remove(file_path)
            print(f'Удален: {file_path}')
        except Exception as e:
            print(f'Ошибка при удалении {file_path}: {e}')


#удаление всех записей об изображениях
try:
    # Удаление таблицы с проверкой на существование
    cursor.execute("DROP TABLE IF EXISTS dataset")
    # Подтверждение изменений
    conn.commit()
    print("Таблица успешно удалена")
except sqlite3.Error as e:
    print(f"Ошибка при удалении таблицы: {e}")
finally:
    # Закрытие соединения
    cursor.close()
    conn.close()