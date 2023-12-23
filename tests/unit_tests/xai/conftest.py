from typing import Tuple

import pandas as pd
import pytest
from sklearn.datasets import make_regression

from prediction.predictor_model import train_predictor_model
from preprocessing.target_encoder import CustomTargetEncoder, train_target_encoder


@pytest.fixture
def target_field_name():
    return "target"


@pytest.fixture
def num_samples():
    return 1000


@pytest.fixture
def transformed_data(target_field_name, num_samples) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Create a regression dataset using sklearn's make_regression.

    Returns:
        Tuple[pd.DataFrame, pd.Series]: A tuple with two elements.
        The first element is a DataFrame with feature values,
        and the second element is a Series with the target values.
    """
    # Create a regression with 2 informative features
    features, targets = make_regression(
        n_samples=num_samples,
        n_features=10,
        n_informative=2,
        random_state=42,
        shuffle=True,
        coef=False
    )
    # Convert to pandas DataFrame and Series
    features_df = pd.DataFrame(
        features, columns=[f"feature_{i}" for i in range(1, features.shape[1] + 1)]
    )
    targets = pd.DataFrame(targets, columns=[target_field_name])
    return features_df, targets


@pytest.fixture
def num_train(num_samples):
    """Get the number of training examples"""
    return int(num_samples * 0.8)


@pytest.fixture
def num_test(num_samples, num_train):
    """Get the number of testing examples"""
    return num_samples - num_train


@pytest.fixture
def transformed_train_and_test_data(transformed_data, num_train, num_test):
    """Fixture to create a sample transformed train DataFrame"""
    features_df, targets = transformed_data
    features_df_train = features_df.head(num_train)
    targets_train = targets.head(num_train)
    features_df_test = features_df.tail(num_test)
    targets_test = targets.tail(num_test)
    return features_df_train, targets_train, features_df_test, targets_test


@pytest.fixture
def transformed_train_inputs(transformed_train_and_test_data):
    """Get training inputs"""
    return transformed_train_and_test_data[0]


@pytest.fixture
def transformed_test_inputs(transformed_train_and_test_data):
    """Get testing inputs"""
    return transformed_train_and_test_data[2]


@pytest.fixture
def predictor(transformed_train_and_test_data, default_hyperparameters):
    # Train a simple model for testing
    features_df_train, targets_train, _, _ = transformed_train_and_test_data
    return train_predictor_model(
        features_df_train, targets_train, default_hyperparameters
    )


@pytest.fixture
def target_encoder(transformed_train_and_test_data, target_field_name):
    """Get the target encoder"""
    _, targets_train, _, _ = transformed_train_and_test_data
    target_encoder = train_target_encoder(
        CustomTargetEncoder(target_field_name), targets_train
    )
    return target_encoder
