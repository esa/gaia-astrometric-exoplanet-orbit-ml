#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file ‘LICENCE.txt’.

from loguru import logger
import sys
import gaia_ad

def set_log_level(log_level: str, log_to_file: bool = False):
    """
    Set the log level for the logger and optionally log to a file.

    Args:
        log_level (str): The log level to set. Options are 'TRACE', 'DEBUG', 'INFO', 'SUCCESS', 'WARNING', 'ERROR', 'CRITICAL'.
        log_to_file (bool): If True, logs will also be saved to a file named 'logfile.log'. Default is False.

    Raises:
        ValueError: If the provided log_level is not one of the expected values.
    """
    # Define valid log levels
    valid_log_levels = ['TRACE', 'DEBUG', 'INFO', 'SUCCESS', 'WARNING', 'ERROR', 'CRITICAL']
    
    # Assert that the provided log_level is valid
    assert log_level.upper() in valid_log_levels, f"Invalid log level: {log_level}. Expected one of {valid_log_levels}."

    # Remove any existing logger configuration
    logger.remove()

    # Add a new logger configuration
    logger.add(
        sys.stderr,
        colorize=True,
        level=log_level.upper(),
        format="<green>{time:HH:mm:ss}</green>|gaia_ad-<blue>{level}</blue>| <level>{message}</level>",
    )

    # Optionally add logging to a file
    if log_to_file:
        logger.add("logfile.log", level=log_level.upper(), format="{time:YYYY-MM-DD HH:mm:ss}|{level}|{message}")

    # Log the new log level
    logger.debug(f"Setting LogLevel to {log_level.upper()}")

    # Set the log level in the gaia_ad module
    gaia_ad.GAIAAD_LOGLEVEL = log_level.upper()
