#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file ‘LICENCE.txt’.
from loguru import logger
import sys

import gaia_ad


def set_log_level(log_level: str):
    """Set the log level for the logger.

    Args:
        log_level (str): The log level to set. Options are 'TRACE','DEBUG', 'INFO', 'SUCCESS', 'WARNING', 'ERROR', 'CRITICAL'.
    """
    logger.remove()
    logger.add(
        sys.stderr,
        colorize=True,
        level=log_level.upper(),
        format="<green>{time:HH:mm:ss}</green>|gaia_ad-<blue>{level}</blue>| <level>{message}</level>",
        # filter="gaia_ad",
    )
    logger.debug(f"Setting LogLevel to {log_level.upper()}")
    gaia_ad.GAIAAD_LOGLEVEL = log_level.upper()
