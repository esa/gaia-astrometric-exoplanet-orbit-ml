#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file ‘LICENCE.txt’.
import warnings

from loguru import logger
import numpy as np
from pyod.models.xgbod import XGBOD
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, precision_recall_curve, auc
from tqdm import tqdm


def predict_on_dataset(
    X: np.ndarray, y: np.ndarray, classifier=XGBOD, n_splits=None, feature_names: list = None
):
    """This method predicts outliers assuming you have some labeled data. You give it a bunch of data
    and labels, and it will train a classifier on it and try to find those presumed "inliers" which aren't
    actually inliers. This is useful for finding outliers in a dataset where you have some labeled data.

    Args:
        X (np.ndarray): The data to be predicted on
        y (np.ndarray): The labels for the data (0 = inlier, 1 = outlier)
        classifier (_type_, optional): Classifier to use, has to provide the functions `fit`,`predict` and `decision_function`. Defaults to XGBOD.
        n_splits (_type_, optional): Number of splits to use in the cross validation. Defaults to # of outliers.
        feature_names (list, optional): List of feature names, used for feature importance. Defaults to None.

    Returns:
        predictions (np.ndarray): The predictions for each data point
        predicted_labels (np.ndarray): The predicted labels for each data point
        clfs (list): List of classifiers used
        importances (list): List of feature importances for each classifier split
        scores (dict): Dictionary containing the scores for the predictions
        split_index_of_sample (np.ndarray): The index of the split for each sample, can be used to identify
        corresponding classifier and importance
    """

    logger.info("Predicting false inliers")

    # Assert data types
    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert isinstance(feature_names, list) or feature_names is None
    # Assert that the data is the same length
    assert len(X) == len(y)
    # Assert that the labels are either 0 or 1
    assert np.all(np.isin(y, [0, 1]))

    test_clf = classifier()
    assert callable(getattr(test_clf, "fit", None)), "Classifier has to have a `fit` method"
    assert callable(getattr(test_clf, "predict", None)), "Classifier has to have a `predict` method"
    assert callable(
        getattr(test_clf, "predict_proba", None)
    ), "Classifier has to have a `predict_proba` method"

    # If n_splits is None, set it to the number of outliers
    if n_splits is None:
        n_splits = np.sum(y)

    logger.debug(f"Using {n_splits} splits")

    # Initialize the cross validation
    skf = StratifiedKFold(n_splits=n_splits)
    skf.get_n_splits(X, y)

    logger.debug("Initialized cross validation")

    # Initialize the predictions and labels
    predictions = np.zeros(len(X)) - 1
    predicted_labels = np.zeros(len(X)) - 1
    split_index_of_sample = np.zeros(len(X)) - 1

    clfs = []
    importances = []
    # If feature names are not given, use the column indices as str
    feature_names = (
        feature_names if feature_names is not None else [str(i) for i in range(X.shape[1])]
    )
    logger.trace(f"Using feature names: {feature_names}")

    logger.debug("Starting cross validation")

    # Store f1 and accuracy scores per split
    f1_scores = []
    acc_scores = []
    aurocs = []
    auc_prc = []

    # Iterate through the splits
    for i, (train_index, test_index) in enumerate(
        tqdm(skf.split(X, y), total=skf.get_n_splits(X, y))
    ):
        # Get the training and testing data
        X_train, X_test = X[train_index], X[test_index]
        y_train, _ = y[train_index], y[test_index]

        # Initialize the classifier
        clf = classifier()

        # Fit the classifier
        clf.fit(X_train, y_train)
        # Get the predictions on test data
        preds = clf.predict(X_test)
        preds_proba = clf.predict_proba(X_test)

        # Compute F1 and accuracy
        f1_scores.append(f1_score(y[test_index], preds))
        acc_scores.append(accuracy_score(y[test_index], preds))

        # Add the labels to the overall labels
        predicted_labels[test_index] = preds
        # Add the predictions to the overall predictions
        if preds_proba.ndim == 1:
            predictions[test_index] = preds_proba
        # Account for different shape returned by classifiers from sklearn
        else:
            predictions[test_index] = preds_proba[:, 1]

        # Add the split number to the overall split number to remember
        split_index_of_sample[test_index] = i

        # Compute AUROC, AUC PRC
        aurocs.append(roc_auc_score(y[test_index], predictions[test_index]))
        precision, recall, _ = precision_recall_curve(y[test_index], predictions[test_index])
        auc_prc.append(auc(recall, precision))

        # Add the classifier to the list of classifiers
        clfs.append(clf)

        # Add the importances to the list of importances
        if isinstance(clf, XGBClassifier):
            clf.get_booster().feature_names = feature_names
            importance = clf.get_booster().get_score(importance_type="weight")
            # Set the importance to 0 for features that are not used
            for feature in feature_names:
                if feature not in importance:
                    importance[feature] = 0

        else:
            try:
                importance = clf.feature_importances_
                importance = dict(zip(feature_names, importance))
            except AttributeError:
                importance = None
                warnings.warn("Classifier has no feature importances, skipping")
        importances.append(importance)

    # Print the average F1 and accuracy scores
    logger.info(
        f"Average F1 score: {np.mean(f1_scores)} +- {np.std(f1_scores)} [min, max] = [{np.min(f1_scores)}, {np.max(f1_scores)}]"
    )
    logger.info(
        f"Average accuracy score: {np.mean(acc_scores)} +- {np.std(acc_scores)}"
        + f"  [min, max] = [{np.min(acc_scores)}, {np.max(acc_scores)}]"
    )
    logger.info(
        f"Average AUROC: {np.mean(aurocs)} +- {np.std(aurocs)}"
        + f"  [min, max] = [{np.min(aurocs)}, {np.max(aurocs)}]"
    )
    logger.info(
        f"Average AUC PRC: {np.mean(auc_prc)} +- {np.std(auc_prc)}"
        + f"  [min, max] = [{np.min(auc_prc)}, {np.max(auc_prc)}]"
    )
    scores = {
        "f1": f1_scores,
        "acc": acc_scores,
        "auroc": aurocs,
        "auc_prc": auc_prc,
    }

    # Assert that we looked at all the data
    assert np.all(predictions != -1)
    assert np.all(predicted_labels != -1)

    # Assert output shapes line up
    assert len(predictions) == len(predicted_labels)
    assert len(predictions) == len(split_index_of_sample)

    # Return the predictions and labels
    return predictions, predicted_labels, clfs, importances, scores, split_index_of_sample
