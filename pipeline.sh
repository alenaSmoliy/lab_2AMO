#!/bin/bash

# Последовательный запуск этапов машинного обучения:
# 1. Создание и разделение данных
# 2. Предобработка (масштабирование)
# 3. Обучение модели
# 4. Оценка качества модели

python data_creation.py
python model_preprocessing.py
python model_preparation.py
python model_testing.py
