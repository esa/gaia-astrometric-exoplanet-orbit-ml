from tqdm import tqdm
import shap
import pandas as pd
import numpy as np
import pickle as pk
import os

from plots import plot_individual_feature_importance


def compute_shap(
    methods, datasets, candidates, features, plot=True, path="results/", only_candidates=True
):
    overall_shap_values = pd.DataFrame(columns=features)
    # We start by loading the clf that classified it as positive and the split_index
    for _, candidate in tqdm(
        candidates.iterrows(),
        total=len(candidates),
        desc="Feature Importance per Candidate Computation",
    ):
        # Create a folder for this candidate in results/feature_importance
        if not os.path.exists(path + str(candidate.source_id)):
            os.makedirs(path + str(candidate.source_id))
        for idx, method in enumerate(methods):
            # Check this candidate was classified positively by this method
            if candidate[method] != 1 and only_candidates:
                continue
            # print("Candidate", candidate.source_id, "was classified as positive by method", method)

            # Load clf and split_index
            if "worst" in method:
                name = method.split("top")[1].partition(".csv")[0]
                split_index = pd.read_csv(
                    "results/split_index" + name + ".csv", dtype={"source_id": np.int64}
                )
            else:
                name = "".join(method.partition("_")[1:]).partition(".csv")[0]
                split_index = pd.read_csv(
                    "results/split_index" + name + ".csv", dtype={"source_id": np.int64}
                )
            # Get split_index of this candidate
            assert candidate.source_id in split_index.source_id.values, (
                "Candidate source_id "
                + str(candidate.source_id)
                + " not found in split_index of method "
                + name
            )
            split_index_candidate = split_index[split_index.source_id == candidate.source_id][
                "split_index"
            ].values[0]
            idx = split_index[split_index.source_id == candidate.source_id].index[0]
            clf = pk.load(open("results/clfs" + name + ".pk", "rb"))
            clf = clf[int(split_index_candidate)]
            X = datasets[method][0]

            # Get the shap tree explainer for this clf
            explainer = shap.TreeExplainer(clf)

            # Get the shap values for this candidate
            # print("----------------------------------")
            # print(method)
            # print("idx", idx)
            # print(X.shape)
            shap_values = explainer(X.iloc[idx : idx + 1])

            # Discard the 0 class component if present
            if len(shap_values.values.shape) == 3:
                shap_values.values = shap_values.values[:, :, 1]

            # print(shap_values)

            # Store the shap values for this candidate
            overall_shap_values.loc[str(candidate.source_id) + "_" + method] = shap_values.values[0]
            # print(overall_shap_values)

            if plot:
                plot_individual_feature_importance(
                    shap_values, explainer, name, candidate, features, path
                )

    return overall_shap_values


def xgb_shap_transform_scale(shap_values, model_prediction):  
    # Compute the transformed base value, 
    # which consists in applying the logit function to the base value
    # Importing the logit function for the base value transformation
    from scipy.special import expit 
    base_value = expit(shap_values.base_values[0])
    
    # Computing the original_explanation_distance to construct the 
    # distance_coefficient later on
    original_explanation_distance = np.sum(shap_values.values, axis=1)
    # print("original_explanation_distance", original_explanation_distance)
    # print("model_prediction", model_prediction)

    # Computing the distance between the model_prediction and the transformed
    # base_value
    distance_to_explain = model_prediction - base_value
    # print("distance_to_explain", distance_to_explain)

    # The distance_coefficient is the ratio between both distances which will 
    # be used later on
    distance_coefficient = original_explanation_distance / distance_to_explain
    # print("distance_coefficient", distance_coefficient)

    # Transforming the original shapley values to the new scale
    shaps = shap_values.values / distance_coefficient

    # Now returning the transformed array
    return shaps,base_value
