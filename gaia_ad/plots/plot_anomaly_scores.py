from loguru import logger
import matplotlib.pyplot as plt


def plot_anomaly_scores(
    scores_by_category: list,
    category_names: list,
    N_bins: int = 10,
    log: bool = True,
    savepath: str = None,
):
    """Creates plot showing distributions of anomaly scores for each category.

    Args:
        scores_by_category (list): List of lists of anomaly scores (float between 0 and 1) for each category.
        category_names (list): List of category names (str).
        N_bins (int, optional): Number of bins in histogram. Defaults to 10.
        log (bool, optional): Whether to use log scale on y-axis. Defaults to True.
        savepath (str, optional): Path to save figure to. Defaults to None.
    """
    logger.info("Plotting anomaly scores...")

    # Check scores are between 0 and 1
    assert all(
        [all([0 <= score <= 1 for score in scores]) for scores in scores_by_category]
    ), "Anomaly scores must be between 0 and 1."

    # Check there is exactly one score for each category
    assert len(scores_by_category) == len(
        category_names
    ), "There must be exactly one score for each category."

    # Create a figure with one subplot per category and one subplot for all categories
    N_plots = len(category_names) + 1
    fig, axs = plt.subplots(N_plots, 1, figsize=(10, 3 * N_plots), sharex=True, dpi=100)

    # Create a histogram with shared binning for all categories
    ax = axs[-1]
    ax.hist(scores_by_category, bins=N_bins, range=(0, 1), log=log, align="left")

    # Remember color per category
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    # Unpack category names and add N per category to plot
    category_names = [
        name + f" (N={len(scores)})" for name, scores in zip(category_names, scores_by_category)
    ]
    plt.legend(category_names)
    plt.xlabel("Score")
    plt.ylabel("# of entries")
    plt.title("All categories")

    # Create a histogram for each category with finer binning
    for scores, name, ax in zip(scores_by_category, category_names, axs[:-1]):
        ax.hist(scores, bins=N_bins * 5, range=(0, 1), log=log, align="left", color=colors.pop(0))
        ax.set_title(name)
        ax.set_ylabel("# of entries")
        ax.set_xlabel("Score")

    # Save figure if savepath is provided
    if savepath is not None:
        plt.savefig(savepath)
        logger.info(f"Saved figure to {savepath}")

    # Show figure
    plt.tight_layout()
    plt.show(block=False)
