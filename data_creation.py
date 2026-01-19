import pandas as pd
import os
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo

# Загружаем датасет "Wine Quality" из UCI Repository
wine_quality = fetch_ucirepo(id=186)

# Извлекаем признаки и целевую переменную
X = wine_quality.data.features
y = wine_quality.data.targets

# Разделяем данные на обучающую и тестовую выборки (70% / 30%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Объединяем признаки и целевую переменную в единые DataFrame'ы
train_df = pd.concat([X_train, y_train], axis=1)
test_df = pd.concat([X_test, y_test], axis=1)

# Создаём директории для сохранения данных, если они ещё не существуют
os.makedirs("train", exist_ok=True)
os.makedirs("test", exist_ok=True)

# Сохраняем обучающую и тестовую выборки в CSV-файлы без индексов
train_df.to_csv("train/train_data.csv", index=False)
test_df.to_csv("test/test_data.csv", index=False)
