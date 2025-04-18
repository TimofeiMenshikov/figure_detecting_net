# Figure Detecting Network

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.6+](https://img.shields.io/badge/Python-3.6%2B-blue.svg)](https://www.python.org/)

Нейронная сеть для обнаружения и классификации геометрических фигур на изображениях. Проект поддерживает распознавание кругов, треугольников, прямоугольников.

## Особенности

- 🎯 Высокая точность на синтетических и реальных изображениях
- �  Поддержка GPU/CPU вычислений
- 📦 Возможность создания своего датасета
- 🔧 Простое расширение на новые классы фигур

## Установка

1. Клонируйте репозиторий:
```bash
git clone https://github.com/TimofeiMenshikov/figure_detecting_net.git
cd figure_detecting_net
```

2. Запустите создание датасета:
```bash
python create_dataset.py
```

3. Запустите обучение модели:
```bash
python net_training.py
```

4. Есть возможность удаления датасета:
```bash
python delete_dataset.py
```

5. Просмотр базы данных
```bash
python test_db.py
```

