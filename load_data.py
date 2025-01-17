import os

from loguru import logger
import pandas as pd


def load_data(
    dataset_version: str = "0.0.1",
    drop_NSS=True,
    make_binary: bool = False,
    clean: bool = True,
    drop_nan_columns=True,
    drop_nan_rows=False,
    label_for_high_confidence_substellar: int = 1,
    path=None,
):
    """Load the specified dataset and apply preprocessing steps.

    Args:
        dataset_version (str, optional): Version of the dataset to load. Defaults to "0.0.1".
        drop_NSS (bool, optional): Whether to drop non-single source. Defaults to True.
        make_binary (bool, optional): Whether to make the labels binary. Defaults to False.
        clean (bool, optional): Whether to clean the dataset. Defaults to True.
        drop_nan_columns (bool, optional): Whether to drop columns with NaN values. Defaults to True.
        drop_nan_rows (bool, optional): Whether to drop rows with NaN values. Defaults to False.
        label_for_high_confidence_substellar (int, optional): Label to use for high confidence substellar candidates,
        they are labeled as "better_substellar_companion_candidates" and likely to be exoplanets or brown dwarfs. Defaults to 1.
        path (str, optional): Path to the data directory. Defaults to data folder in gaia_orbit_class.

    Returns:
        pandas.DataFrame: Dataframe containing the features.
        pandas.DataFrame: Dataframe containing the labels.
        pandas.DataFrame: Dataframe containing the original labels.
        pandas.DataFrame: Dataframe containing the full dataset.
        pandas.DataFrame: Dataframe containing the non-candidates.
        dict: Dictionary mapping labels to integers.
        pandas.DataFrame: Dataframe containing the source ids.
    """

    available_versions = ["0.0.1", "0.0.2", "0.0.3", "0.0.4", "0.0.6"]

    datapath = os.path.dirname(os.path.abspath(__file__)) + "/data/"
    if path is not None:
        datapath = path

    # Check the file exists
    assert os.path.exists(datapath), f"Looked for dataset at {datapath} but it does not exist"

    assert (
        dataset_version in available_versions
    ), f"dataset_version must be one of {available_versions}"

    # Only row or column can be dropped at a time
    assert not (
        drop_nan_columns and drop_nan_rows
    ), "Only one of drop_nan_columns or drop_nan_rows can be True"

    logger.info(f"Loading dataset version {dataset_version} from path {datapath}")
    # Load respective dataset
    if dataset_version in ["0.0.1", "0.0.2", "0.0.3", "0.0.4", "0.0.6"]:
        df, labels = _load_data_0_0_1(datapath)

    # Drop non-single source
    if drop_NSS:
        candidates = df[df["label"].notna()]  # Drop NSS that are not candidates
    else:
        candidates = df

    # Eliminating false_positive_orbit labeled ones
    logger.info(
        f"Eliminating {len(candidates[candidates['label'] == 'false_positive_orbit'])} false positive orbits"
    )
    candidates = candidates[candidates["label"] != "false_positive_orbit"]
    df = df[df["label"] != "false_positive_orbit"]

    # Reset indices
    candidates.reset_index(drop=True, inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Keep original labels
    y = candidates.label.copy()
    source_ids = candidates.source_id.copy()
    # Fill NaN values with "non_single_source"
    y.fillna("non_single_source", inplace=True)
    y_original = y.copy().values

    logger.info(f"Loaded {len(candidates)} candidates")

    # Convert to int labels
    if make_binary:
        label_mapping = {
            "exoplanet": 1,
            "brown_dwarf_companion": 1,
            "substellar_companion_candidates": 0,
            "very_low_mass_stellar_companion": 0,
            "binary_star": 0,
            "non_single_source": 0,
            "better_substellar_companion_candidates": label_for_high_confidence_substellar,
        }
        y.replace(label_mapping, inplace=True)

    else:
        label_mapping = {
            "binary_star": 0,
            "very_low_mass_stellar_companion": 0,
            "substellar_companion_candidates": 1,
            "brown_dwarf_companion": 2,
            "exoplanet": 3,
            "non_single_source": 4,
            "better_substellar_companion_candidates": label_for_high_confidence_substellar,
        }
        y.replace(label_mapping, inplace=True)

    logger.info(f"Loaded {len(y)} labels")
    logger.info(f"Class distribution: {y.value_counts()}")

    logger.info(
        f"Cleaning dataset... Original Size: {len(candidates)} with {len(candidates.columns)} features"
    )

    # Clean up the feature dataframe
    # forget labels
    X = candidates.drop(columns=["label"])

    # Investigate NaN distribution
    logger.debug("Number of rows with NaN entries: ", X.isnull().any(axis=1).sum())
    logger.debug("Number of NaN entries: ", X.isnull().sum().sum())
    # Print which columns have how many NaNs (skpping cols with no NaNs)
    logger.debug("Number of NaNs per column: ")
    logger.debug(X.isnull().sum()[X.isnull().sum() > 0])

    # Print source id of rows that appear twice in X
    idx = X[X.duplicated(keep=False)].index.tolist()
    if len(idx) > 0:
        logger.warning("Rows that appear twice in X:")
        logger.warning(set(df.iloc[idx].source_id.tolist()))
        logger.warning(len(set(df.iloc[idx].source_id.tolist())))

    if clean:
        # drop non-numeric columns
        X = X.drop(
            columns=[
                "source_id",
                "astrometric_primary_flag",
                "nss_solution_type",
                "reference",
                "id",
            ]
        )
        # drop columns with confounders
        X = X.drop(
            columns=[
                "parallax_agis",
                "parallax_nss",
                "phot_g_mean_mag",
                "phot_bp_mean_mag",
                "phot_rp_mean_mag",
                "pmdec_agis",
                "pmdec_nss",
                "pmra_agis",
                "pmra_nss",
            ]
        )
        # drop columns with NaN values
        if drop_nan_columns:
            logger.debug(
                "Dropping columns with NaN values...Size before: {}".format(len(X.columns))
            )
            X = X.dropna(axis=1)
            logger.debug("Size after: {}".format(len(X.columns)))
        # drop rows with NaN values
        if drop_nan_rows:
            logger.debug("Dropping rows with NaN values...Size before: {}".format(len(X)))
            X = X.dropna(axis=0)
            logger.debug("Size after: {}".format(len(X)))

    non_candidates = df[~df.index.isin(labels.index)]
    # Drop columns from non-candidates that were dropped in X
    non_candidates = non_candidates[X.columns]

    logger.info(f"Cleaned dataset... Size: {len(X)} with {len(X.columns)} features")

    # Assert the length of all variables match
    assert len(X) == len(y)
    assert len(X) == len(y_original)
    assert len(X) == len(source_ids)
    if not drop_NSS:
        assert len(X) == len(df)
    # assert max(y) <= len(label_mapping) - 1  # 0-indexed
    assert max(y.index) == len(y) - 1  # 0-indexed
    # Assert all source_ids are in df
    assert len(source_ids[~source_ids.isin(df.source_id)]) == 0

    return X, y, y_original, df, non_candidates, label_mapping, source_ids


def _load_data_0_0_1(datapath):
    """Load the dataset version 0.0.1.

    Args:
        datapath (str): Path to the data directory.

    Returns:
        pandas.DataFrame: Dataframe containing the data.
        pandas.DataFrame: Dataframe containing the labels.
    """

    # Load all dataframes
    single_source_df = pd.read_parquet(datapath + "gaia_source_astrometric_orbits.parquet")
    non_single_source_df = pd.read_parquet(
        datapath + "nss_two_body_orbit_astrometric_orbits.parquet"
    )
    labels = pd.read_parquet(datapath + "labelled_sources.parquet")

    # Combine the two dataframes (nss & single source)
    merged_df = single_source_df.merge(
        non_single_source_df, on="source_id", suffixes=("_agis", "_nss")
    )
    # Set label where available
    merged_df = merged_df.merge(labels, on="source_id", how="left")

    return merged_df, labels
