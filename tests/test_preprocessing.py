import pandas as pd

from src.preprocessing import (
    TRAIN_FEATURES_FILE,
    TEST_FEATURES_FILE,
    TRAIN_TARGET_FILE,
    TEST_TARGET_FILE,
    SCALER_FILE,
)


def test_training_features_exist():
    assert TRAIN_FEATURES_FILE.exists()


def test_testing_features_exist():
    assert TEST_FEATURES_FILE.exists()


def test_training_target_exists():
    assert TRAIN_TARGET_FILE.exists()


def test_testing_target_exists():
    assert TEST_TARGET_FILE.exists()


def test_scaler_exists():
    assert SCALER_FILE.exists()


def test_training_feature_shape():
    df = pd.read_csv(TRAIN_FEATURES_FILE)

    assert df.shape == (404, 13)


def test_testing_feature_shape():
    df = pd.read_csv(TEST_FEATURES_FILE)

    assert df.shape == (102, 13)


def test_training_target_shape():
    df = pd.read_csv(TRAIN_TARGET_FILE)

    assert df.shape == (404, 1)


def test_testing_target_shape():
    df = pd.read_csv(TEST_TARGET_FILE)

    assert df.shape == (102, 1)