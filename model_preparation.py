import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib

# Загружаем обучающий набор данных (уже масштабированный)
train_data = pd.read_csv('train/train_data_scaled.csv', sep=',')

# Отделяем признаки от целевой переменной
X_train = train_data.drop(columns=["quality"])
y_train = train_data["quality"]

# Инициализируем и обучаем модель случайного леса
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)

# Сохраняем обученную модель в файл для последующего использования
joblib.dump(rf_model, 'trained_model.pkl')
