import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
import json

# Загружаем тестовые данные
test_data = pd.read_csv('test/test_data.csv')

# Отделяем признаки от целевой переменной
X_test = test_data.drop(columns=["quality"])
y_test = test_data["quality"]

# Загружаем ранее обученную модель
model = joblib.load('trained_model.pkl')

# Выполняем предсказания на тестовой выборке
y_pred = model.predict(X_test)

# Вычисляем метрики качества модели
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

# Выводим среднюю абсолютную ошибку
print(f"Средняя абсолютная ошибка (MAE) на тестовых данных: {mae:.4f}")

# Сохраняем MAE в JSON-файл
with open("testing_mae.json", "w", encoding="utf-8") as json_file:
    json.dump({"mae": mae}, json_file, ensure_ascii=False, indent=4)
