import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import pandas as pd

# Чтение данных из файла Excel
df = pd.read_excel('1.xlsx', header=None)

# Получение значений из столбцов A и B
x = df.iloc[:, 0].values
y = df.iloc[:, 1].values

# Преобразование в массивы numpy
x = np.array(x)
y = np.array(y)

def scatterplot(x, y, threshold=3):
    # Рассчитываем среднее и стандартное отклонение для каждого временного ряда
    mean_x, std_x = np.mean(x), np.std(x)
    mean_y, std_y = np.mean(y), np.std(y)

    # Определяем границы для выбросов
    x_lower_bound = mean_x - threshold * std_x
    x_upper_bound = mean_x + threshold * std_x
    y_lower_bound = mean_y - threshold * std_y
    y_upper_bound = mean_y + threshold * std_y

    # Индексы выбросов
    outliers_indices = np.where((x < x_lower_bound) | (x > x_upper_bound) | (y < y_lower_bound) | (y > y_upper_bound))[
        0]

    # Восстановление значений (для простоты - средние)
    x_reconstructed = np.copy(x)
    y_reconstructed = np.copy(y)
    x_reconstructed[outliers_indices] = mean_x
    y_reconstructed[outliers_indices] = mean_y

    return x_reconstructed, y_reconstructed

def dbscan(x, y, eps=0.3, min_samples=2):
    # Объединение векторов в один массив
    data = np.column_stack((x, y))

    # DBSCAN
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    dbscan.fit(data)

    # Индексы выбросов
    outliers_indices = np.where(dbscan.labels_ == -1)[0]

    # Восстановление значений (для простоты - средние)
    x_reconstructed = np.copy(x)
    y_reconstructed = np.copy(y)
    x_reconstructed[outliers_indices] = np.mean(x)
    y_reconstructed[outliers_indices] = np.mean(y)

    return x_reconstructed, y_reconstructed

# Обнаружение выбросов и восстановление значений с помощью диаграммы рассеяния
x_reconstructed_scatter, y_reconstructed_scatter = scatterplot(x, y)
# Обнаружение выбросов и восстановление значений с помощью пространственной кластеризации (DBSCAN)
x_reconstructed_dbscan, y_reconstructed_dbscan = dbscan(x, y)

# Визуализация результатов
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.scatter(x, y, label='Original Data')
plt.scatter(x_reconstructed_scatter, y_reconstructed_scatter, label='Reconstructed Data (Scatter)', color='red',
            marker='x')
plt.title('Scatter Plot Reconstruction')
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(x, y, label='Original Data')
plt.scatter(x_reconstructed_dbscan, y_reconstructed_dbscan, label='Reconstructed Data (DBSCAN)', color='green',
            marker='o')
plt.title('DBSCAN Reconstruction')
plt.legend()

plt.tight_layout()
plt.show()