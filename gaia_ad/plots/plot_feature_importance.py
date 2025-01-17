from loguru import logger
from matplotlib import pyplot as plt
import pandas as pd


def plot_feature_importance(
    importances: pd.DataFrame,
    savepath: str = None,
):
    """Creates plot showing feature importance of all classifiers.

    Args:
        importances (pd.DataFrame): DataFrame with feature importances for each classifier.
        savepath (str, optional): Path to save figure to. Defaults to None.
    """
    logger.info("Plotting feature importance...")

    assert isinstance(importances, pd.DataFrame), "Importances must be a pandas DataFrame"
    # Check data shape
    assert importances.shape[0] > 0, "Importances must have at least one row"
    assert importances.shape[1] > 0, "Importances must have at least one column"

    # Fill NaNs with 0
    importances = importances.fillna(0)

    # Sort the dataframe by mean importance
    importances = importances.reindex(importances.mean().sort_values(ascending=True).index, axis=1)

    # Compute mean and std
    mean = importances.mean()
    std = importances.std()

    # Plot bars with error bars for std and set legend
    _ = plt.figure(figsize=(12, 8))
    plt.barh(
        importances.columns, mean, xerr=std, align="center", alpha=0.5, ecolor="black", capsize=10
    )

    # Set axis labels
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")
    # Set title
    plt.title("Feature importance of all models")
    # Describe std in legend
    plt.legend(["Mean +- Std"])

    # Save figure if savepath is provided
    if savepath is not None:
        plt.savefig(savepath, dpi=100, bbox_inches="tight")
        logger.info(f"Saved figure to {savepath}")

    # Show figure
    plt.tight_layout()
    plt.show(block=False)
