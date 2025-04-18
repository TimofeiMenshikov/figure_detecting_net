import torch.nn as nn
import torch


class LinearImageClassifier(nn.Module):
    def __init__(self, input_size=64*64*3, num_classes=10):
        super().__init__()
        self.flatten = nn.Flatten()
        self.layers = nn.Sequential(

            nn.Linear(256 + 32, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

        self.image_layer = nn.Sequential(
            nn.Linear(input_size, 256),        # Для изображений 64x64 после пулинга
            nn.ReLU(),
            nn.Dropout(0.2)
        )

            # Дополнительный слой для обработки признака "количество углов"
        self.feature_layer = nn.Sequential(
            nn.Linear(1, 32),  # 1 вход (количество углов)
            nn.ReLU()
        )

    def forward(self, img, features):
        img = self.flatten(img)  # Преобразует [batch, 3, 64, 64] → [batch, 3*64*64]
        img = self.image_layer(img)

        features = self.feature_layer(features.float())

        data  = torch.cat([img, features], dim=1)

        data = self.layers(data)

        return data