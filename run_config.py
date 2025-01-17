# Ignore flake E402 in this file
# flake8: noqa: E402

from imblearn.ensemble import BalancedRandomForestClassifier
import xgboost as xgb
import pandas as pd
import numpy as np
import pickle as pk

from load_data import load_data
from gaia_ad.supervised.predict_on_dataset import predict_on_dataset


def run_config(
    classifier,
    data,
    label_for_better_ssc,
    N_compare,
    cand_threshold,
    dataset_version,
    testing=False,
):
    np.random.seed(0)
    # Set classifier
    if classifier == "xgb":
        clf = lambda: xgb.XGBClassifier(
            learning_rate=0.1,
            max_depth=3,
            objective="binary:logistic",
            reg_lambda=1,
            base_score=0.5,
            random_state=0,
        )
    elif classifier == "rf":
        clf = lambda: BalancedRandomForestClassifier(
            max_depth=2, random_state=0, n_estimators=100, n_jobs=-1, replacement=False
        )

    # Load data
    (X, y, y_original, df, non_candidates, label_mapping, source_ids) = load_data(
        dataset_version=dataset_version,
        drop_NSS=(data == "ssc"),
        make_binary=True,
        clean=True,
        drop_nan_columns=False,
        drop_nan_rows=False,
        label_for_high_confidence_substellar=label_for_better_ssc,
        path="data/" + dataset_version + "/",
    )

    # Drop most data if testing, but make sure to keep at least 1 candidate
    if testing:
        N_test = 1000
        X = X.iloc[:N_test]
        y = y.iloc[:N_test]
        y_original = y_original[:N_test]
        y[:10] = 1
        y_original[:10] = 1
        y.index = range(len(y))
        source_ids = source_ids[:N_test]
        df = df[df.source_id.isin(source_ids)]

    # Compute and store anomaly scores
    n_splits = 3 if testing else (y == 1).sum()
    predictions, predicted_labels, clfs, importances, scores, split_index = predict_on_dataset(
        X.values, y.values, classifier=clf, n_splits=n_splits, feature_names=X.columns.tolist()
    )

    # Make sure source_ids and predictions are numpy arrays of the same length and all in df
    assert len(source_ids) == len(predictions)
    assert df["source_id"].dtype == source_ids.dtype

    # Create a DataFrame from source_ids and predictions
    predictions_df = pd.DataFrame(
        {
            "source_id": source_ids,
            "predictions": predictions,
            "predicted_labels": predicted_labels,
            "y": y,
            "y_original": y_original,
        }
    )
    assert predictions_df.source_id.isin(df.source_id).sum() == len(predictions_df), (
        f"Not all source_ids were matched successfully {sum(predictions_df.source_id.isin(df.source_id))} out of {len(predictions_df)}"
        + f" source_ids were matched successfully. Unmatched were"
        + str(predictions_df[~predictions_df.source_id.isin(df.source_id)].source_id.values)
    )
    print("#######################")
    print(predictions_df)

    # Merge df and predictions_df and make sure all source_ids were matched successfully
    df = df.merge(predictions_df, on="source_id", how="left")

    # Identify top N candidates based on anomaly score and them being prospectively exoplanets
    # these are label 0 but we know they are not exoplanets
    excluded_y_labels = [
        "very_low_mass_stellar_companion",
        "binary_star",
    ]
    top_candidates = df[
        (df.predictions > cand_threshold) & (df.y == 0) & (~df.y_original.isin(excluded_y_labels))
    ].sort_values(by="predictions", ascending=False)
    top_candidates = top_candidates.head(N_compare)

    # Print source_id, prediction, label of them sorted by the anomaly score
    print("###### TOP CANDIDATES ######")
    print(top_candidates)

    # Also accumulate top without restrictions as a sanity check
    top_candidates_without_filter = df[(df.predictions > cand_threshold)].sort_values(
        by="predictions", ascending=False
    )
    top_candidates_without_filter = top_candidates_without_filter.head(N_compare)
    print("###### TOP CANDIDATES WITHOUT FILTER DISTRIBUTION ######")
    print(top_candidates_without_filter.y_original.value_counts())

    all_exo_bd = df[df.y_original.isin(["exoplanet", "brown_dwarf_companion"])].sort_values(
        by="predictions", ascending=False
    )

    # Save top N_compare suspected outliars in the substellar candidates to a csv file and scores for method
    method = classifier + "_" + data + "_" + "label_for_better_ssc=" + str(label_for_better_ssc)
    top_candidates.to_csv("results/top_" + method + ".csv", index=False)
    top_candidates_without_filter.to_csv("results/no_filter_top_" + method + ".csv", index=False)
    all_exo_bd.to_csv("results/worst_truepositive_top_" + method + ".csv", index=False)
    # Save score dict
    np.save("results/scores_" + method + ".npy", scores)
    # Save clfs with pickle for later use
    with open("results/clfs_" + method + ".pk", "wb") as f:
        pk.dump(clfs, f)
    # Save split_index for later use, create a df with source_id & split_index
    split_df = pd.DataFrame({"source_id": source_ids, "split_index": split_index})
    split_df.to_csv("results/split_index_" + method + ".csv", index=False)
