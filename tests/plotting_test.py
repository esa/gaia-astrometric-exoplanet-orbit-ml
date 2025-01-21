#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file ‘LICENCE.txt’.
"""Tests for the plotting functions."""

from gaia_ad.plots.plot_anomaly_scores import plot_anomaly_scores
from gaia_ad.plots.plot_tsne import plot_tsne
from gaia_ad.plots.plot_feature_importance import plot_feature_importance

import numpy as np
import pandas as pd


def test_anomaly_score_plot():
    """Test the anomaly score plot."""
    # Seed the test
    np.random.seed(42)

    # Create some random data
    test_categories = ["Cat1", "Cat2", "Cat3"]
    test_scores = []
    for cat in test_categories:
        some_random_scores = np.random.normal(
            loc=np.random.rand(),
            scale=np.random.rand() * 0.2,
            size=np.random.randint(5, 1000),
        )
        # Clip scores to be between 0 and 1
        some_random_scores = np.clip(some_random_scores, 0, 1)
        test_scores.append(some_random_scores)

    plot_anomaly_scores(
        scores_by_category=test_scores, category_names=test_categories, N_bins=10, log=False
    )


def test_tsne_plot():
    """Test the tsne plot."""

    # set_log_level("DEBUG")

    # Seed the test
    np.random.seed(42)

    # Create some random data
    test_categories = ["Cat1", "Cat2", "Cat3"]
    test_scores = []
    for cat in test_categories:
        some_random_scores = np.random.normal(
            loc=np.random.rand(),
            scale=np.random.rand() * 0.2,
            size=np.random.randint(2, 3000),
        )
        # Clip scores to be between 0 and 1
        some_random_scores = np.clip(some_random_scores, 0, 1)
        test_scores.append(some_random_scores)

    # Create some random 3d data for the tsne plot
    underlying_data = []
    for idx, cat in enumerate(test_categories):
        N_data = len(test_scores[idx])
        some_random_data = np.random.normal(
            loc=np.random.rand(),
            scale=np.random.rand() * 0.2,
            size=(N_data, 3),
        )
        underlying_data.append(some_random_data)

    plot_tsne(
        underlying_data=underlying_data,
        category_names=test_categories,
        scores_by_category=test_scores,
        n_iter=1000,
        dynamic_sizing=True,
    )


def test_feat_importance_plot():
    importances = [
        {"a": 1, "b": 2, "c": 3, "d": 4},
        {"a": 4, "b": 3, "c": 2, "d": 1},
        {"a": 1, "b": 2, "c": 3, "d": 4},
    ]
    df = pd.DataFrame(importances)
    plot_feature_importance(df)
