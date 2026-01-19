import pandas as pd
from sklearn.preprocessing import StandardScaler

# Загружаем исходные обучающие и тестовые данные
train_data = pd.read_csv('train/train_data.csv')
test_data = pd.read_csv('test/test_data.csv')

# Отделяем признаки от целевой переменной ("quality")
X_train = train_data.drop(columns=["quality"])
y_train = train_data["quality"]
X_test = test_data.drop(columns=["quality"])
y_test = test_data["quality"]

# Применяем стандартизацию: обучаем скалер на обучающих данных и трансформируем обе выборки
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Восстанавливаем DataFrame с сохранением исходных имён столбцов
train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)
train_scaled_df["quality"] = y_train.values

test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns)
test_scaled_df["quality"] = y_test.values

# Сохраняем стандартизированные данные в CSV-файлы без индексов
train_scaled_df.to_csv('train/train_data_scaled.csv', index=False)
test_scaled_df.to_csv('test/test_data_scaled.csv', index=False)
