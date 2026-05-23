# src/evaluate.py

import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split

from src.preprocess import prepare_data


def evaluate_model():

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

    # График фактических и предсказанных значений

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

    plt.savefig(
        'reports/figures/predictions.png'
    )

    plt.show()

    # Feature importance

    model_step = model.named_steps['model']

    if hasattr(model_step, 'feature_importances_'):

        importance = pd.DataFrame({
            'feature': features_test.columns,
            'importance': model_step.feature_importances_
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

        plt.title('Feature Importance')

        plt.tight_layout()

        plt.savefig(
            'reports/figures/feature_importance.png'
        )

        plt.show()


if __name__ == '__main__':

    evaluate_model()