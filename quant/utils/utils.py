#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

"""Common utility functions."""

from typing import Any


def noop(*args: Any, **kwargs: Any) -> None:
    """No-op that returns None."""
    return None
