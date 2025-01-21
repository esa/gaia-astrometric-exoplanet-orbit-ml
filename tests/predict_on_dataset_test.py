#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file ‘LICENCE.txt’.
"""Test for the predict_on_dataset function."""

from gaia_ad.supervised.predict_on_dataset import predict_on_dataset
import gaia_ad
import numpy as np
import pytest
import xgboost as xgb
from imblearn.ensemble import BalancedRandomForestClassifier


# Ignore warning about missing feature importance
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_predict_on_dataset():
    """Test the function a small toy dataset to ensure it works."""
    gaia_ad.set_log_level("TRACE")

    # Seed the test
    np.random.seed(42)

    # Create some random data
    X = np.random.normal(size=(1000, 10))
    y = np.zeros(len(X))
    # Set a random set of points to be outliers
    y[np.random.choice(len(X), size=10, replace=False)] = 1

    # Predict on the dataset
    predictions, predicted_labels, clfs, feat_importance, scores, splits = predict_on_dataset(
        X, y, n_splits=2
    )

    # Check the results
    assert len(predictions) == len(X)
    assert len(predicted_labels) == len(X)
    assert predicted_labels.shape == y.shape
    assert predictions.shape == y.shape
    assert np.all(predictions != -1)
    assert np.all(predicted_labels != -1)
    assert np.all(np.isin(predicted_labels, [0, 1]))
    assert np.all(predictions > 0)
    assert np.all(predictions < 1)
    assert len(clfs) == 2
    assert len(feat_importance) == 2
    assert feat_importance[0] is None
    assert feat_importance[1] is None

    # Create some random data
    X = np.random.normal(size=(1000, 4))
    y = np.zeros(len(X))
    y[np.random.choice(len(X), size=10, replace=False)] = 1

    # Test with XGBClassifier
    clf = lambda: xgb.XGBClassifier(
        learning_rate=0.1, max_depth=3, objective="binary:logistic", reg_lambda=1, base_score=0.5
    )
    predictions, predicted_labels, clfs, feat_importance, scores, splits = predict_on_dataset(
        X, y, n_splits=3, classifier=clf
    )

    # Check the results
    assert len(predictions) == len(X)
    assert len(predicted_labels) == len(X)
    assert np.all(predictions != -1)
    assert np.all(predicted_labels != -1)
    assert np.all(np.isin(predicted_labels, [0, 1]))
    assert np.all(predictions > 0)
    assert np.all(predictions < 1)
    assert len(clfs) == 3
    assert len(feat_importance) == 3
    # Feature importance should work for XGBClassifier
    assert all([len(feat_importance[i]) == 4 for i in range(len(feat_importance))])

    clf = lambda: BalancedRandomForestClassifier(
        max_depth=2, random_state=0, n_estimators=100, n_jobs=-1, replacement=False
    )
    predictions, predicted_labels, clfs, feat_importance, scores, splits = predict_on_dataset(
        X, y, n_splits=2, classifier=clf, feature_names=["a", "b", "c", "d"]
    )

    # Check the results
    assert len(predictions) == len(X)
    assert len(predicted_labels) == len(X)
    assert np.all(predictions != -1)
    assert np.all(predicted_labels != -1)
    assert np.all(np.isin(predicted_labels, [0, 1]))
    assert np.all(predictions > 0)
    assert np.all(predictions < 1)
    assert len(clfs) == 2
    assert len(feat_importance) == 2
    # Feature importance should work for XGBClassifier
    assert all([len(feat_importance[i]) == 4 for i in range(len(feat_importance))])
