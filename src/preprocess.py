# src/preprocess.py

import pandas as pd
import os


def load_data(path):
    """
    Загрузка исходных данных
    """

    data = pd.read_csv(
        path,
        index_col=[0],
        parse_dates=[0]
    )

    # Сортировка по времени
    data = data.sort_index()

    return data


def resample_data(data):
    """
    Агрегация количества заказов по часу
    """

    data = data.resample('1h').sum()

    return data


def create_features(data):
    """
    Создание признаков для временного ряда
    """

    # Календарные признаки
    data['dayofweek'] = data.index.dayofweek
    data['hour'] = data.index.hour

    # One-Hot Encoding часов
    data = data.join(
        pd.get_dummies(
            data['hour'],
            drop_first=True,
            prefix='hour'
        )
    )

    # Удаляем исходный признак hour
    data.drop('hour', axis=1, inplace=True)

    # Лаговые признаки
    for lag in [1, 23, 24, 48, 72, 168]:
        data[f'lag_{lag}'] = data['num_orders'].shift(lag)

    # Скользящее среднее за 24 часа
    data['rolling_mean'] = (
        data['num_orders']
        .shift()
        .rolling(24)
        .mean()
    )

    # Удаление пропусков
    data = data.dropna()

    return data


def save_processed_data(data, path):
    """
    Сохранение обработанных данных
    """

    os.makedirs(
        os.path.dirname(path),
        exist_ok=True
    )

    data.to_csv(path)


def prepare_data(input_path, output_path=None):
    """
    Полный ETL/preprocessing pipeline
    """

    # Extract
    data = load_data(input_path)

    # Transform
    data = resample_data(data)

    data = create_features(data)

    # Load
    if output_path:
        save_processed_data(data, output_path)

    return data


if __name__ == '__main__':

    data = prepare_data(
        input_path='data/raw/taxi.csv',
        output_path='data/processed/taxi_features.csv'
    )

    print(data.head())

    print('\nShape:', data.shape)