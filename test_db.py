import sqlite3

# Подключение к БД
conn = sqlite3.connect('dataset/dataset.db')
cursor = conn.cursor()


cursor.execute("SELECT * FROM dataset")
print("all figures:", cursor.fetchall())