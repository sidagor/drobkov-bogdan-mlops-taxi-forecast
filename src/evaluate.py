# src/evaluate.py

import os
import joblib
import mlflow
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split

from src.preprocess import prepare_data


def evaluate_model():

    # Создание папок

    os.makedirs(
        'reports/figures',
        exist_ok=True
    )

    # Загрузка данных

    data = prepare_data(
        input_path='data/raw/taxi.csv'
    )

    # Train/Test split

    train, test = train_test_split(
        data,
        shuffle=False,
        test_size=0.1
    )

    # Features / Target

    features_test = test.drop(
        'num_orders',
        axis=1
    )

    target_test = test['num_orders']

    # Загрузка модели

    model = joblib.load(
        'models/best_model.pkl'
    )

    # Предсказания

    predictions = model.predict(
        features_test
    )

    # RMSE

    rmse = root_mean_squared_error(
        target_test,
        predictions
    )

    print(f'RMSE: {rmse:.2f}')

    # MLflow logging

    with mlflow.start_run(run_name='evaluation'):

        mlflow.log_metric(
            'rmse',
            rmse
        )

        # График предсказаний

        plt.figure(figsize=(15, 6))

        plt.plot(
            target_test.index,
            target_test,
            label='Real'
        )

        plt.plot(
            target_test.index,
            predictions,
            label='Predictions'
        )

        plt.title('Taxi Orders Forecast')

        plt.xlabel('Date')

        plt.ylabel('Orders')

        plt.legend()

        plt.tight_layout()

        predictions_path = (
            'reports/figures/predictions.png'
        )

        plt.savefig(
            predictions_path
        )

        mlflow.log_artifact(
            predictions_path
        )

        plt.show()

        # Feature importance

        model_step = model.named_steps['model']

        if hasattr(
            model_step,
            'feature_importances_'
        ):

            importance = pd.DataFrame({
                'feature': features_test.columns,
                'importance': (
                    model_step.feature_importances_
                )
            })

            importance = importance.sort_values(
                by='importance',
                ascending=False
            )

            plt.figure(figsize=(10, 8))

            plt.barh(
                importance['feature'][:15],
                importance['importance'][:15]
            )

            plt.title(
                'Feature Importance'
            )

            plt.tight_layout()

            importance_path = (
                'reports/figures/feature_importance.png'
            )

            plt.savefig(
                importance_path
            )

            mlflow.log_artifact(
                importance_path
            )

            plt.show()


if __name__ == '__main__':

    evaluate_model()