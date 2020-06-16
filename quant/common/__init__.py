#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

"""Common utilities and infrastructure for Quant."""

import logging


def init_logging(log_level: str) -> None:
    """
    Initialize the logger.

    Args:
        log_level (str): logging level, e.g. DEBUG, INFO, WARNING.
    """
    level_map = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
    }
    logging.basicConfig(level=level_map[log_level])
