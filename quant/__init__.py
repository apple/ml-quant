#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

"""A toolkit supporting binary quantization of neural networks."""

from typing import Dict, Optional
from typing_extensions import Protocol

from quant.common.metrics import Metric

__version__ = '0.2.0'


# Define some common types here

MetricDict = Dict[str, Metric]


class Hook(Protocol):
    """Hook protocol."""

    def __call__(
        self, epoch: int, global_step: int,
        log_interval: int = 10, values_dict: Optional[dict] = None
    ) -> None:
        """Define function signature for a hook."""
        ...
