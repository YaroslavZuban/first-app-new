import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import pandas as pd
from pykalman import KalmanFilter

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

def kalman_filter(x, y, process_noise=1e-5, measurement_noise=1e-2):
    # Объединение векторов в один массив
    data = np.column_stack((x, y))

    # Инициализация фильтра Калмана
    kf = KalmanFilter(initial_state_mean=data[0],
                      initial_state_covariance=np.eye(data.shape[1]),
                      observation_covariance=measurement_noise * np.eye(data.shape[1]),
                      transition_covariance=process_noise * np.eye(data.shape[1]))

    # Прогон данных через фильтр Калмана
    smoothed_state_means, _ = kf.filter(data)

    # Извлечение отфильтрованных значений
    x_filtered, y_filtered = smoothed_state_means[:, 0], smoothed_state_means[:, 1]

    return x_filtered, y_filtered

def exponential_moving_average(x, y, alpha=0.2):
    # Создание DataFrame для удобства работы с pandas
    df_smoothed = pd.DataFrame({'x': x, 'y': y})

    # Применение экспоненциального скользящего среднего к каждому столбцу
    df_smoothed['x_smoothed'] = df_smoothed['x'].ewm(alpha=alpha, adjust=False).mean()
    df_smoothed['y_smoothed'] = df_smoothed['y'].ewm(alpha=alpha, adjust=False).mean()

    return df_smoothed['x_smoothed'].values, df_smoothed['y_smoothed'].values

# Обнаружение выбросов и восстановление значений с помощью диаграммы рассеяния
x_reconstructed_scatter, y_reconstructed_scatter = scatterplot(x, y)
# Обнаружение выбросов и восстановление значений с помощью пространственной кластеризации (DBSCAN)
x_reconstructed_dbscan, y_reconstructed_dbscan = dbscan(x, y)
# Применение фильтра Калмана к данным после обнаружения выбросов с использованием scatterplot
x_filtered_kalman, y_filtered_kalman = kalman_filter(x_reconstructed_scatter, y_reconstructed_scatter)
# Сглаживание значений ряда с помощью алгоритма экспоненциального скользящего среднего
x_smoothed_ema, y_smoothed_ema = exponential_moving_average(x_filtered_kalman, y_filtered_kalman, alpha=0.2)

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

plt.figure(figsize=(10, 5))
plt.scatter(x, y, label='Original Data')
plt.scatter(x_reconstructed_scatter, y_reconstructed_scatter, label='Reconstructed Data (Scatter)', color='red',
            marker='x')
plt.scatter(x_filtered_kalman, y_filtered_kalman, label='Filtered Data (Kalman)', color='blue', marker='^')
plt.title('Kalman Filtered Data (after Scatterplot Reconstruction)')
plt.legend()

plt.figure(figsize=(10, 5))
plt.scatter(x, y, label='Original Data')
plt.scatter(x_smoothed_ema, y_smoothed_ema, label='Smoothed Data (EMA)', color='purple', marker='s')
plt.title('Exponential Moving Average Smoothing')
plt.legend()

plt.tight_layout()
plt.show()