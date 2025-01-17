import pandas as pd
import numpy as np


def create_occurence_df(methods, N_compare):
    method_names = []
    top = pd.DataFrame()
    # Create DF to store all unique sources identified as candidates
    candidates = pd.DataFrame(
        columns=[
            "source_id",
            "label",
            "cumulative_prediction",
            "occurences",
            "occurences_ssc",
            "occurences_nss",
            *methods,
        ]
    )
    # Set source_id dtype to int64 to avoid float conversion
    candidates["source_id"] = np.zeros(N_compare, dtype=int)

    for method in methods:
        if "no_filter" in method or "worst_truepositive" in method:
            name = (
                method.split("_")[3]
                + "_"
                + method.split("_")[4]
                + "_"
                + method.split("for_")[1][:12]
            )
            dataset = method.split("_")[4]
        else:
            name = (
                method.split("_")[1]
                + "_"
                + method.split("_")[2]
                + "_"
                + method.split("for_")[1][:12]
            )
            dataset = method.split("_")[2]
        assert dataset in ["nss", "ssc"], "Dataset is not nss or ssc, but " + str(dataset)
        method_names.append(name)
        method_result = pd.read_csv(method, dtype={"source_id": np.int64})
        # For all entries of method results check that the source_id is int
        assert (
            method_result.dtypes["source_id"] == int
            or method_result.dtypes["source_id"] == np.int64
        ), "source_id is not int, but " + str(method_result.dtypes["source_id"])
        top[name] = method_result.source_id.values

        # Add to candidates DF based on source_id
        # Count occurences and cumulative predictions
        for row in method_result.itertuples():
            if row.source_id in candidates.source_id.values:
                candidates.loc[
                    candidates.source_id == row.source_id, "cumulative_prediction"
                ] += row.predictions
                candidates.loc[candidates.source_id == row.source_id, "occurences"] += 1

                # Keep track of occurences in ssc and nss datasets
                if dataset == "ssc":
                    candidates.loc[candidates.source_id == row.source_id, "occurences_ssc"] += 1
                elif dataset == "nss":
                    candidates.loc[candidates.source_id == row.source_id, "occurences_nss"] += 1

                # Set label of this method to 1 for this candidate
                candidates.loc[candidates.source_id == row.source_id, method] = 1
            else:
                assert isinstance(row.source_id, int), "source_id is not int64, but " + str(
                    type(row.source_id)
                )
                # Adding new candidate using concat
                new_candidate = pd.DataFrame(
                    [
                        [
                            row.source_id,
                            row.label,
                            row.predictions,
                            1,
                            int(dataset == "ssc"),  # 1 if dataset is ssc, 0 otherwise
                            int(dataset == "nss"),  # 1 if dataset is nss, 0 otherwise
                            *np.zeros(len(methods)),  # Add zeros for all individual methods
                        ]
                    ],
                    columns=[
                        "source_id",
                        "label",
                        "cumulative_prediction",
                        "occurences",
                        "occurences_ssc",
                        "occurences_nss",
                        *methods,
                    ],
                )
                new_candidate[method] = 1
                candidates = pd.concat(
                    [
                        candidates,
                        new_candidate,
                    ]
                )
                assert (
                    method_result.dtypes["source_id"] == int
                    or method_result.dtypes["source_id"] == np.int64
                ), "source_id is not int, but " + str(candidates.dtypes["source_id"])

    # Convert relative occurence. divide by number of runs
    total_runs = len(methods)

    candidates["relative_occurence"] = candidates.occurences / total_runs
    candidates["relative_occurence_ssc"] = candidates.occurences_ssc / (total_runs / 2)
    candidates["relative_occurence_nss"] = candidates.occurences_nss / (total_runs / 2)
    # Candidates with bssc label could only be found in ssc label = 0 runs
    # So we double their occurence
    candidates.loc[
        candidates.label == "better_substellar_companion_candidates", "relative_occurence"
    ] *= 2
    candidates.loc[
        candidates.label == "better_substellar_companion_candidates", "relative_occurence_ssc"
    ] *= 2
    candidates.loc[
        candidates.label == "better_substellar_companion_candidates", "relative_occurence_nss"
    ] *= 2

    # Same for candidates from nss (not ssc) dataset
    candidates.loc[candidates.label == "non_single_source", "relative_occurence"] *= 2

    # Sort by relative occurence
    candidates = candidates.sort_values(by="relative_occurence", ascending=False)
    # Print top N_compare candidates
    print("#######################")
    print("#######################")
    print("Top accumulated candidates are")
    print(candidates.head(N_compare))
    # Remove rows source_id == 0
    candidates = candidates[candidates.source_id != 0]

    print("#######################")
    print("candidates are")
    print(candidates)
    print("Top sources are")
    print(top)
    return candidates, top, method_names
