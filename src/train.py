# src/train.py

import joblib
import mlflow
import mlflow.sklearn
import os

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    TimeSeriesSplit
)

from sklearn.metrics import root_mean_squared_error

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from catboost import CatBoostRegressor

from src.preprocess import prepare_data


def train_models():

    # Загрузка и preprocessing

    data = prepare_data(
        input_path='data/raw/taxi.csv',
        output_path='data/processed/taxi_features.csv'
    )

    # Train/Test split

    train, test = train_test_split(
        data,
        shuffle=False,
        test_size=0.1
    )

    print(f'Train size: {len(train)}')
    print(f'Test size: {len(test)}')

    os.makedirs("models", exist_ok=True)

    os.makedirs(
        "catboost_info",
        exist_ok=True
    )

    # Features / Target

    features_train = train.drop('num_orders', axis=1)
    target_train = train['num_orders']

    features_test = test.drop('num_orders', axis=1)
    target_test = test['num_orders']

    # TimeSeries CV

    tscv = TimeSeriesSplit(n_splits=5)

    # Модели и параметры

    models = {

        'LinearRegression': {
            'model': LinearRegression(),
            'params': {}
        },

        'RandomForest': {
            'model': RandomForestRegressor(
                random_state=42
            ),

            'params': {
                'model__n_estimators': [100, 200],
                'model__max_depth': [5, 10]
            }
        },

        'CatBoost': {
            'model': CatBoostRegressor(
                verbose=0,
                random_state=42,
                train_dir="catboost_info",
                allow_writing_files=True
            ),

            'params': {
                'model__depth': [4, 6],
                'model__learning_rate': [0.03, 0.1]
            }
        }
    }

    # Поиск лучшей модели

    best_model = None
    best_rmse = float('inf')
    best_model_name = None

    for name, config in models.items():

        print(f'\nTraining {name}...')

        with mlflow.start_run(run_name=name):

            # Pipeline

            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('model', config['model'])
            ])

            # GridSearchCV

            grid_search = GridSearchCV(
                estimator=pipeline,
                param_grid=config['params'],
                cv=tscv,
                scoring='neg_root_mean_squared_error',
                n_jobs=-1
            )

            # Обучение

            grid_search.fit(
                features_train,
                target_train
            )

            # Лучшая модель

            model = grid_search.best_estimator_

            # Предсказания

            predictions = model.predict(
                features_test
            )

            # RMSE

            rmse = root_mean_squared_error(
                target_test,
                predictions
            )

            print(f'{name} RMSE: {rmse:.2f}')

            print(
                'Best params:',
                grid_search.best_params_
            )

            # MLflow logging

            mlflow.log_metric(
                'rmse',
                rmse
            )

            mlflow.log_params(
                grid_search.best_params_
            )

            mlflow.sklearn.log_model(
                model,
                'model'
            )

            # Выбор лучшей модели

            if rmse < best_rmse:

                best_rmse = rmse
                best_model = model
                best_model_name = name

    # Итог

    print(f'\nBest model: {best_model_name}')
    print(f'Best RMSE: {best_rmse:.2f}')

    # Сохранение модели
    
    joblib.dump(
        best_model,
        'models/best_model.pkl'
    )

    print('\nModel saved to models/best_model.pkl')

    return best_model, best_rmse


if __name__ == '__main__':

    train_models()