#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file ‘LICENCE.txt’.
# Ignore flake E402 in this file
# flake8: noqa: E402
import warnings

warnings.filterwarnings("ignore")
from glob import glob

import pandas as pd
import numpy as np

import gaia_ad

gaia_ad.set_log_level("WARNING")

from load_data import load_data

from plots import (
    plot_shap_scatters,
    plot_scores_and_overlap,
    plot_shap_violin,
)
from create_occurence_df import create_occurence_df
from run_config import run_config
from compute_shap import compute_shap

# Make results directory if not exists
import os

if not os.path.exists("results"):
    os.makedirs("results")

plot = True
plot_individual_feature_importance = True
rerun = True

##### Configuration ########
classifiers = ["rf", "xgb"]
# classifiers = ["rf"]
data = ["nss", "ssc"]
dataset_version = "0.0.6"
# data = ["ssc"]
label_for_better_ssc = [0, 1]  # 1 considers them as anomalies, 0 considers them as normal
N_compare = 50  # Top how many to compare
cand_threshold = 0.001  # Threshold for candidates


print("#######################")
print("#######################")
print("Starting candidates computation")
print("#######################")
print("#######################")

if rerun:
    for classifier in classifiers:
        for dataset in data:
            for label in label_for_better_ssc:
                # Fix seed
                np.random.seed(0)

                print("Running for classifier: ", classifier)
                print("Running for dataset: ", dataset)
                print("Running for label: ", label)

                run_config(
                    classifier,
                    dataset,
                    label,
                    N_compare,
                    cand_threshold,
                    dataset_version,
                    testing=False,
                )

print("#######################")
print("#######################")
print("Finished candidates computation")
print("#######################")
print("#######################")

# Consolidate results
##### Below code compares between different methods
#####
methods = glob("results/top*")
no_filter_methods = glob("results/no_filter*")
worst_cand_methods = glob("results/worst_truepositive_*")
methods.sort()
no_filter_methods.sort()
worst_cand_methods.sort()

candidates, top, method_names = create_occurence_df(methods, N_compare)
candidates_no_filter, _, _ = create_occurence_df(no_filter_methods, N_compare)
worst_truepositive_candidates, _, _ = create_occurence_df(worst_cand_methods, N_compare)

# Save candidates to csv
candidates.to_csv("results/candidates.csv", index=False)
candidates_no_filter.to_csv("results/candidates_no_filter.csv", index=False)
worst_truepositive_candidates.to_csv("results/worst_truepositive.csv", index=False)

# Also create a pd with the scores for each method
results = glob("results/scores*")
results.sort()
example_score = np.load(results[0], allow_pickle=True).item()
scores = pd.DataFrame(index=example_score.keys())
stds = pd.DataFrame(index=example_score.keys())
# print("Found results:", results)
for result, name in zip(results, method_names):
    # # Doublecheck method name
    # print(
    #     name, result.split("_")[1] + "_" + result.split("_")[2] + "_" + result.split("for_")[1][:12]
    # )
    assert (
        name
        == result.split("_")[1] + "_" + result.split("_")[2] + "_" + result.split("for_")[1][:12]
    )
    score = np.load(result, allow_pickle=True).item()
    scores[name] = [np.mean(score[key]) for key in score.keys()]
    stds[name] = [np.std(score[key]) for key in score.keys()]
    # print("Score for method", name, "is", scores[name].values)
# print("Scores are")
# print(scores)
scores.to_csv("results/all_scores.csv")
plot_scores_and_overlap(scores, stds, top, method_names)

# For each of the top N candidates, we look at all models that classified it as an anomaly
# We need to get the datasets again for this
datasets = {}
gaia_ad.set_log_level("WARNING")  # Avoid spam from load_data
for method in methods:
    X, y, y_original, df, non_candidates, label_mapping, _ = load_data(
        dataset_version=dataset_version,
        drop_NSS=method.split("_")[2] == "ssc",
        make_binary=True,
        clean=True,
        drop_nan_columns=False,
        drop_nan_rows=False,
        label_for_high_confidence_substellar=method.split(".csv")[0][-1],
        path="data/" + dataset_version + "/",
    )
    datasets[method] = (X, y, y_original, df, non_candidates, label_mapping)

features = datasets[methods[0]][0].columns.tolist()

overall_shap_values = compute_shap(
    methods,
    datasets,
    candidates,
    features,
    plot=plot_individual_feature_importance,
    path="results/shap_candidates/",
)

# Compute decision plots for each exoplanets and bds
# reindex dataset to worst_cand_methods
datasets_worst = {}
for worst_method, old_key, value in zip(worst_cand_methods, methods, datasets.values()):
    datasets_worst[worst_method] = value
shap_exo = compute_shap(
    worst_cand_methods,
    datasets_worst,
    worst_truepositive_candidates,
    features,
    plot=plot_individual_feature_importance,
    path="results/shap_exo/",
    only_candidates=False,
)

# Plot the overall shap values as violin plot
if plot:
    plot_shap_violin(overall_shap_values, features)

X_big, _, _, df_big, _, _, _ = load_data(
    dataset_version=dataset_version,
    drop_NSS=False,
    make_binary=True,
    clean=True,
    drop_nan_columns=False,
    drop_nan_rows=False,
    label_for_high_confidence_substellar=0,
    path="data/" + dataset_version + "/",
)

# For top three features also plot the shap scatter
if plot:
    # Compute the top three features
    top_features = overall_shap_values.abs().mean().sort_values(ascending=False).head(15).index
    for feature in top_features:
        plot_shap_scatters(feature, overall_shap_values, X_big, df_big)

print("#######################")
print("#######################")
print("Script finished successfully")
print("#######################")
print("#######################")
