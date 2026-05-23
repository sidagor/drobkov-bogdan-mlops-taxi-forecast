# tests/test_preprocess.py

from src.preprocess import prepare_data


def test_no_missing_values():

    data = prepare_data(
        input_path='data/raw/taxi.csv'
    )

    assert data.isnull().sum().sum() == 0


def test_features_created():

    data = prepare_data(
        input_path='data/raw/taxi.csv'
    )

    expected_features = [
        'dayofweek',
        'lag_1',
        'lag_24',
        'rolling_mean'
    ]

    for feature in expected_features:

        assert feature in data.columns