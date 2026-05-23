# tests/test_model.py

import os
import joblib

from src.train import train_models
from src.preprocess import prepare_data

from sklearn.model_selection import train_test_split


def test_model_rmse():

    _, rmse = train_models()

    assert rmse < 48


def test_model_saved():

    train_models()

    assert os.path.exists(
        'models/best_model.pkl'
    )


def test_prediction_shape():

    model, _ = train_models()

    data = prepare_data(
        input_path='data/raw/taxi.csv'
    )

    train, test = train_test_split(
        data,
        shuffle=False,
        test_size=0.1
    )

    features_test = test.drop(
        'num_orders',
        axis=1
    )

    predictions = model.predict(
        features_test
    )

    assert len(predictions) == len(features_test)