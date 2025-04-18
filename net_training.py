from torchvision import datasets, transforms
import torch
import os
import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from torch.utils.data import random_split
import queue
import threading

import sqlite3


# local import 
from net_architecture import LinearImageClassifier
from visualisation    import RealTimePlot


def get_accuracy(dataloader, model):

    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, features, labels in dataloader:

            inputs, features, labels = inputs.to(device), features.to(device), labels.to(device)
            outputs = model(inputs, features)

            _, predicted = torch.max(outputs.data, 1)

            for i in range(len(predicted)):
                correct += (predicted[i] == labels[i])
                total   += 1

    return float(100 * correct / total)


### Dataset and Dataloader
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

])


class SQLImageDataset(Dataset):
    def __init__(self, db_path, image_dir, transform=None):
        self.db_path = db_path
        self.image_dir = image_dir  # Путь к папке с изображениями
        self.transform = transform
        self.data = []


        self.label_map = {
            "circle": 0,
            "rectangle": 1,
            "triangle": 2,
            # Добавьте все классы из вашей задачи
        }
        
        # Загружаем данные из базы при инициализации
        self._load_data()

    def _load_data(self):
        """Загрузить все записи из таблицы features."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT id, num_verticles, figure_type FROM dataset")
        sql_data = cursor.fetchall()

        self.data = sql_data

        cursor.close()
        conn.close()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Получаем запись из базы
        record = self.data[idx]
        id_, num_verticles, figure_type_str = record

        figure_type = self.label_map[figure_type_str]
        
        # Формируем путь к изображению
        image_path = f"{self.image_dir}/shape_{id_}.png"
        
        # Загружаем изображение
        try:
            image = Image.open(image_path).convert("RGB")
        except FileNotFoundError:
            raise ValueError(f"Image {image_path} not found!")
        
        # Применяем трансформы
        if self.transform:
            image = self.transform(image)
        
        # Преобразуем признаки и метку в тензоры
        features = torch.tensor([num_verticles], dtype=torch.float32)
        figure_type = torch.tensor(figure_type, dtype=torch.long)
        
        return image, features, figure_type


dataset = SQLImageDataset("dataset/dataset.db", "dataset", transform = transform)


n_train = int(len(dataset) * 0.64)
n_valid = int(len(dataset) * 0.16)
n_test  = len(dataset) - n_train - n_valid

train_dataset, valid_dataset, test_dataset = random_split(
    dataset,
    [n_train, n_valid, n_test],
    generator=torch.Generator().manual_seed(42)  # Для воспроизводимости
)

train_dataloader = DataLoader(
    train_dataset,
    batch_size=10,
    shuffle = True
)

valid_dataloader = DataLoader(
    valid_dataset,
    batch_size = 10,
    shuffle = True
)

test_dataloader = DataLoader(
    test_dataset,
    batch_size = 10,
    shuffle = True
)


### model 

#Параметры модели
input_size = (3, 64, 64)  # [каналы, высота, ширина]
num_classes = 3           # круг, квадрат, треугольник

model = LinearImageClassifier(num_classes=num_classes)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
print(f"Используется устройство: {device}")

loss_fn = nn.CrossEntropyLoss()  # Для классификации
optimizer = optim.Adam(model.parameters(), lr=0.0001)  # Или SGD


train_accs = []
valid_accs = []



def net_training(model, num_epochs, train_losses_queue, valid_losses_queue):

    for epoch in range(num_epochs):

        # Обучение
        model.train()

        train_loss = 0.0

        for batch in train_dataloader:
            inputs, features, labels = batch
            inputs, features, labels = inputs.to(device), features.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs, features)
            loss = loss_fn(outputs, labels)
            train_loss += loss.item()

            loss.backward()
            optimizer.step()

        train_acc = get_accuracy(train_dataloader, model)
        print(f"Train Loss: {train_loss/len(train_dataloader):.4f}", "Train accuracy: ", train_acc)
        

        train_losses_queue.put(train_loss/len(train_dataloader))
        train_accs.append(train_acc)


        # Валидация
        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            for val_batch in valid_dataloader:
                inputs, features, labels = val_batch
                inputs, features, labels = inputs.to(device), features.to(device), labels.to(device)
                outputs = model(inputs, features)
                val_loss += loss_fn(outputs, labels).item()

            valid_acc = get_accuracy(valid_dataloader, model)

            print(f"Val Loss: {val_loss/len(valid_dataloader):.4f}", "Valid accuracy: ", valid_acc)
            
            valid_losses_queue.put(val_loss/len(valid_dataloader))
            valid_accs.append(valid_acc)

        print("Test accuracy: ", get_accuracy(test_dataloader, model))

    # Сохранение
    torch.save(model.state_dict(), "model.pth")



train_losses_queue = queue.Queue()
valid_losses_queue = queue.Queue()


num_epochs = 50

net_training_thread = threading.Thread(
    target = net_training,
    args   = (model, num_epochs, train_losses_queue, valid_losses_queue),
    daemon = True
)

net_training_thread.start()


plotter = RealTimePlot(train_losses_queue, valid_losses_queue, 2)
plotter.start()

