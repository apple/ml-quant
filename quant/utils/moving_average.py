#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

"""Exponential moving average layer."""

import torch
import torch.nn as nn


class MovingAverage(nn.Module):
    """Exponential moving average."""

    def __init__(self, momentum: torch.Tensor) -> None:
        """
        Construct moving average layer.

        Args:
            momentum: A vector indicating the momentum to use for the corresponding row.
        """
        super(MovingAverage, self).__init__()
        self.register_buffer('num_batches_tracked', torch.tensor(0))
        self.register_buffer('momentum', momentum)
        self.register_buffer('moving_average', torch.tensor([0.0] * len(momentum)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        """Return the current moving average, given a vector x."""
        if self.training:
            with torch.no_grad():
                if self.num_batches_tracked.item() > 0:  # type: ignore
                    old = self.momentum * self.moving_average  # type: ignore
                    new = (torch.ones_like(self.momentum) - self.momentum) * x  # type: ignore
                    self.moving_average.copy_(old + new)  # type: ignore
                else:
                    self.moving_average.copy_(x)  # type: ignore
                self.num_batches_tracked += 1  # type: ignore

        return self.moving_average  # type: ignore
