from loguru import logger
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.manifold import TSNE


def plot_tsne(
    underlying_data: list,
    category_names: list,
    scores_by_category: list = None,
    n_iter: int = 2000,
    dynamic_sizing: bool = True,
    default_point_size=20,
    savepath: str = None,
    random_state: int = 42,
):
    """Creates plot showing t-SNE projection of data with color-coded categories and size-coded anomaly scores.

    Args:
        underlying_data (np.ndarray): Data for t-SNE projection.
        category_names (list): List of category names (str).
        scores_by_category (list): List of lists of anomaly scores (float between 0 and 1) for each category. Used for hue.
        n_iter (int, optional): Number of iterations for t-SNE. Defaults to 2000.
        dynamic_sizing (bool, optional): Whether to dynamically size point size in plot. Defaults to True.
        default_point_size (int, optional): Default point size in plot. Defaults to 20.
        savepath (str, optional): Path to save figure to. Defaults to None.
        random_state (int, optional): Random state for t-SNE. Defaults to 42.
    """
    logger.info("Plotting t-SNE projection...")

    if scores_by_category is not None:
        # Check scores are between 0 and 1
        assert all(
            [all([0 <= score <= 1 for score in scores]) for scores in scores_by_category]
        ), "Anomaly scores must be between 0 and 1."

        # Check there is exactly one score for each category
        assert len(scores_by_category) == len(
            category_names
        ), "There must be exactly one score for each category."
        use_scores = True
    else:
        # Create dummy scores
        scores_by_category = [[0.5] * len(data) for data in underlying_data]
        use_scores = False

    # Check there is exactly one score for each category
    assert len(underlying_data) == len(
        category_names
    ), "There must be exactly one score for each category."

    # Check the number of dimensions is the same for all data and matches the number of scores
    assert all(
        [data.shape[1] == underlying_data[0].shape[1] for data in underlying_data]
    ), "All data must have the same number of dimensions."
    assert all(
        [data.shape[0] == len(scores) for data, scores in zip(underlying_data, scores_by_category)]
    ), "All data must have the same number of entries as the number of scores."

    # Check types
    assert isinstance(underlying_data, list), "underlying_data must be a list"
    for data in underlying_data:
        assert isinstance(data, np.ndarray), "underlying_data must be a list of numpy arrays"

    # Combine everything into a dataframe
    df = []
    for category, scores, data in zip(category_names, scores_by_category, underlying_data):
        # Split data into individual dimensions
        data_per_dimension = np.split(data, data.shape[1], axis=1)

        data_dict = {}
        data_dict["Category"] = [category] * len(data)
        data_dict["Score"] = scores
        for i, dim in enumerate(data_per_dimension):
            data_dict[f"Data_col{i}"] = dim.flatten()
        df.append(data_dict)

    logger.debug(f"Dataframe: {df}")

    df = pd.concat([pd.DataFrame(d) for d in df])
    logger.debug(f"Dataframe: {df}")

    # Compute t-SNE projection
    tsne = TSNE(n_components=2, n_iter=n_iter, verbose=1, random_state=random_state)
    # Ignore the category and score columns in the t-SNE projection
    X = df[[col for col in df.columns if col not in ["Category", "Score"]]]
    X_tsne = tsne.fit_transform(X)

    # Add t-SNE projection to a dataframe
    tsne_df = pd.DataFrame()
    tsne_df["Category"] = df["Category"]
    tsne_df["Score"] = df["Score"]
    tsne_df["Component 1"] = X_tsne[:, 0]
    tsne_df["Component 2"] = X_tsne[:, 1]
    logger.debug(f"t-SNE dataframe: {tsne_df}")

    # Plot t-SNE projection, colored by category
    for category in category_names:
        if dynamic_sizing:
            s = 100 if len(tsne_df[tsne_df["Category"] == category]) < 100 else default_point_size
        else:
            s = default_point_size
        sns.scatterplot(
            data=tsne_df[tsne_df["Category"] == category],
            x="Component 1",
            y="Component 2",
            alpha=1.0,
            s=s,
        ).set(title="t-SNE projection (color-coded by category)")

    plt.legend(category_names)

    # Save figure if savepath is provided
    if savepath is not None:
        plt.savefig(savepath + "_category.png", dpi=150)

    plt.show(block=False)

    # Plot t-SNE projection, colored by score
    if use_scores:
        if dynamic_sizing:
            s = 100 if len(tsne_df) < 100 else default_point_size
        else:
            s = default_point_size
        sns.scatterplot(
            data=tsne_df,
            x="Component 1",
            y="Component 2",
            hue="Score",
            palette="viridis",
            alpha=1.0,
            s=s,
        ).set(title="t-SNE projection (color-coded by score)")

        # Save figure if savepath is provided
        if savepath is not None:
            plt.savefig(savepath + "_score.png", dpi=150)

        plt.show(block=False)
