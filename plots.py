import matplotlib.pyplot as plt
import numpy as np
import shap
from sklearn.metrics import ConfusionMatrixDisplay
import pandas as pd


def plot_shap_violin(overall_shap_values, features):
    plt.figure(figsize=(15, 5), dpi=300)
    shap.plots.violin(overall_shap_values.values, feature_names=list(features), show=False)
    plt.tight_layout()
    plt.savefig("results/shap_candidates/overall.png")
    plt.close()


def plot_shap_scatters(feature, overall_shap_values, X, df):
    # For each feature plot the scatter of the shap values
    # against the histogram of the distribution
    # in the same figure
    # using matplotlib instead of shap
    ax = plt.figure(figsize=(15, 5), dpi=300).add_subplot(111)
    ax2 = ax.twinx()
    source_ids = overall_shap_values.index.str.split("_").str[0].astype(np.int64)
    # print(source_ids)
    # for each source_id get the corresponding index in X
    data_idx = [df[df.source_id == source_id].index[0] for source_id in source_ids]
    # print(data_idx)
    # Get the X values for this feature allowing duplicate rows
    X_candidates = X.iloc[data_idx]
    # print(X_candidates)
    # Plot the shap values as scatter
    ax2.scatter(
        X_candidates[feature].values,
        overall_shap_values[feature].values,
        alpha=1.0,
    )

    # Using transparent alpha to make the histogram in the background
    # Create log bins if feature is mass_function_msun or radial_velocity_error
    if feature == "mass_function_msun" or feature == "radial_velocity_error":
        bins = np.logspace(
            np.log10(X[feature].min()),
            np.log10(X[feature].max()),
            100,
        )
    else:
        bins = 100

    ax.hist(X[feature].values, bins=bins, alpha=0.25)

    # if feature is mass_function_msun or radial_velocity_error we make the x axis logarithmic
    if feature == "mass_function_msun" or feature == "radial_velocity_error":
        ax.set_xscale("log")
        ax2.set_xscale("log")
    ax.set_xlabel(feature)
    ax.set_ylabel("Histogram")
    ax2.set_ylabel("SHAP values")
    plt.tight_layout()
    plt.savefig("results/shap_candidates/" + feature + ".png")
    plt.close()


def plot_combined_individual_feature_importance(
    source_id, shap_values, base_values, data, features, model_names, path
):
    # Also plot the decision plot
    plt.figure(figsize=(15, 5))
    features = [feature + "=" + f"{val:.2e}" for feature, val in zip(features, data[0][0])]
    shap.multioutput_decision_plot(
        base_values,
        shap_values,
        0,
        feature_names=features,
        features=data,
        feature_display_range=slice(-1, -11, -1),
        legend_labels=model_names,
        legend_location="lower right",
        show=False,
        title="Decision plot for Gaia DR3 " + str(source_id),
        plot_color="coolwarm",
        xlim=(0, 1) if "RandomForest" in model_names[0] else None,
        link="identity" if "RandomForest" in model_names[0] else "logit",
    )
    plt.tight_layout()
    plt.savefig(path + "_multi_decision.pdf")
    plt.close()


def plot_individual_feature_importance(shap_values, explainer, name, candidate, features, path):
    # Plot the shap values as bar plot
    plt.figure(figsize=(15, 5), dpi=300)
    shap.plots.bar(shap_values, show=False, show_data=True)
    plt.tight_layout()
    plt.savefig(path + str(candidate.source_id) + "/" + name + ".png")
    plt.close()

    # Also plot the decision plot
    plt.figure(figsize=(15, 5))
    exp = explainer.expected_value
    if isinstance(exp, list) or isinstance(exp, np.ndarray):
        exp = exp[1]
    print("exp", exp)
    print("shap_values.values", shap_values.values)
    print("shap_values.data", shap_values.data)
    shap.decision_plot(
        exp,
        shap_values.values,
        feature_names=features,
        show=False,
        features=shap_values.data,
        # feature_display_range=slice(-1, -11, -1),
        feature_order="importance",
    )
    plt.tight_layout()
    plt.savefig(path + str(candidate.source_id) + "/" + name + "_decision.png")
    plt.close()


def plot_scores_and_overlap(scores, stds, top, method_names):
    # Plot scores as bar chart for each metric with error bars with dpi=300
    fig = plt.figure(figsize=(15, 5), dpi=300)
    scores.plot(
        kind="bar",
        figsize=(15, 5),
        title="Scores for each method [+-std]",
        yerr=stds,
        ax=fig.gca(),
        fontsize=16,
    )
    plt.ylim(0, 1.1)
    # Annotate bottom left corner with bad and top left corner with good
    # with large font
    plt.annotate(
        "Bad",
        xy=(0.0, -0.025),
        xytext=(0.0, 0.0),
        xycoords="axes fraction",
        textcoords="offset points",
        ha="left",
        va="top",
        fontsize=20,
    )
    plt.annotate(
        "Good",
        xy=(0.0, 1.025),
        xytext=(0.0, 0.0),
        xycoords="axes fraction",
        textcoords="offset points",
        ha="left",
        va="bottom",
        fontsize=20,
    )
    plt.tight_layout()
    plt.savefig("results/figure_scores.png")
    plt.close()

    # Compute pairwise overlaps between methods
    overlaps = pd.DataFrame(index=method_names, columns=method_names)
    for method in method_names:
        for method2 in method_names:
            overlaps.loc[method, method2] = len(set(top[method]).intersection(set(top[method2])))

    # Plot overlaps color coded via confusion matrix
    matrix = overlaps.values.astype(float)
    disp = ConfusionMatrixDisplay(matrix, display_labels=overlaps.index)
    disp.plot(cmap=plt.cm.Blues)
    # Make x axis labels vertical
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig("results/figure_overlaps.png")
    plt.close()
