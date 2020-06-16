#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

"""Activation quantization."""

from abc import abstractmethod
from enum import Enum
from typing import List, Tuple

import torch
import torch.nn as nn

import quant.binary.quantization as quantization
from quant.utils.moving_average import MovingAverage


class MovingAverageMode(Enum):
    """
    Mode for moving average.

    See :class:`~quant.binary.activation_quantization.ActivationQuantizer`.
    """

    off = 'off'
    eval_only = 'eval_only'
    train_and_eval = 'train_and_eval'


class ActivationQuantizer(nn.Module):
    """
    Activation quantizer abstract class.

    The moving average mode can have 3 options: 'off', 'eval_only', or 'train_and_eval'.

    When moving_average_mode is 'off', moving average is not used.

    When moving_average_mode is 'eval_only', the moving average is tracked but not used
    during training and only used during evaluation mode.

    When moving_average_mode is 'train_and_eval' the moving average is tracked and applied
    during training and used during evaluation as well.

    Currently, 'train_and_eval' can only be used with a single GPU
    and does not support ``nn.DataParallel``.

    The momentum is a value in [0, 1] used in exponential moving average update.
    If the momentum is `alpha`, the update function is:
    `alpha * x + (1 - alpha) * x_new`
    """

    def __init__(
        self,
        num_scaling_factors: int,
        moving_average_mode: str = 'off',
        moving_average_momentum: float = 0.99,
    ) -> None:
        """Construct an activation quantizer."""
        super(ActivationQuantizer, self).__init__()

        momentum_vec = [moving_average_momentum] * num_scaling_factors

        self.num_scaling_factors = num_scaling_factors
        self.moving_avg_module = MovingAverage(torch.tensor(momentum_vec))
        self.moving_average_mode = MovingAverageMode(moving_average_mode)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        """Forward pass of quantizing activation."""
        if self.training:
            # batch_vs is a 2D tensor that stores each v_i along each row
            batch_vs, x_q = self._batch_quantization(x)

            if self.moving_average_mode != MovingAverageMode.off:
                vs_batch_avg = batch_vs.mean(1)
                # Calling moving_avg_module will update its internal statistics under the hood.
                # This is similar to the forward pass of batch norm.
                moving_avg_vs = self.moving_avg_module(vs_batch_avg)

                if self.moving_average_mode == MovingAverageMode.train_and_eval:
                    # If we want to use the scalars with moving average, we need to expand
                    # every scaling factor tensor to the batch size from a single mean element.
                    vs = [
                        moving_avg_vs[i].expand(x.shape[0])
                        for i in range(self.num_scaling_factors)
                    ]

                    x_q = self._moving_average_quantization(x, vs)
        else:
            if self.moving_average_mode != MovingAverageMode.off:
                # If we want to use the scalars with moving average, we need to expand
                # every scaling factor tensor to the batch size from a single mean element.
                vs = [
                    self.moving_avg_module.moving_average[i].expand(x.shape[0])  # type: ignore
                    for i in range(self.moving_avg_module.moving_average.size(0))  # type: ignore
                ]

                x_q = self._moving_average_quantization(x, vs)
            else:
                batch_vs, x_q = self._batch_quantization(x)

        return x_q

    @abstractmethod
    def _batch_quantization(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return a 2-tuple of (scaling factors, quantized x)."""
        raise NotImplementedError  # pragma: no cover

    @abstractmethod
    def _moving_average_quantization(
        self, x: torch.Tensor, vs: List[torch.Tensor]
    ) -> torch.Tensor:
        """Return quantized x using vs."""
        raise NotImplementedError  # pragma: no cover


class ActivationQuantizerLS1(ActivationQuantizer):
    """Activation quantizer using least squares, 1 bit."""

    def __init__(
        self,
        moving_average_mode: str = 'off',
        moving_average_momentum: float = 0.99,
    ) -> None:
        """Construct an activation quantizer using least squares with 1 bit."""
        super(ActivationQuantizerLS1, self).__init__(
            1, moving_average_mode, moving_average_momentum
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        """Forward pass of quantizing activation using least squares 1 bit."""
        return super(ActivationQuantizerLS1, self).forward(x)

    def _batch_quantization(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return a 2-tuple of (scaling factors, quantized x)."""
        batch_v1, x_q = quantization.quantizer_ls_1(x)
        return batch_v1.view(1, -1), x_q

    def _moving_average_quantization(
        self, x: torch.Tensor, vs: List[torch.Tensor]
    ) -> torch.Tensor:
        """Return quantized x using vs."""
        v1 = vs[0]
        _, x_q = quantization.quantizer_ls_1(x, v1)
        return x_q


class ActivationQuantizerLS2(ActivationQuantizer):
    """Activation quantizer using least squares, 2 bits."""

    def __init__(
        self,
        moving_average_mode: str = 'off',
        moving_average_momentum: float = 0.99,
    ) -> None:
        """Construct an activation quantizer using least squares with 2 bit."""
        super(ActivationQuantizerLS2, self).__init__(
            2, moving_average_mode, moving_average_momentum
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        """Forward pass of quantizing activation using least squares 2 bits."""
        return super(ActivationQuantizerLS2, self).forward(x)

    def _batch_quantization(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return a 2-tuple of (scaling factors, quantized x)."""
        batch_v1, batch_v2, x_q = quantization.quantizer_ls_2(x)
        return torch.stack([batch_v1, batch_v2]), x_q

    def _moving_average_quantization(
        self, x: torch.Tensor, vs: List[torch.Tensor]
    ) -> torch.Tensor:
        """Return quantized x using vs."""
        v1, v2 = vs[0], vs[1]
        _, _, x_q = quantization.quantizer_ls_2(x, v1, v2)
        return x_q


class ActivationQuantizerLST(ActivationQuantizer):
    """Activation quantizer using least squares, ternary."""

    def __init__(
        self,
        moving_average_mode: str = 'off',
        moving_average_momentum: float = 0.99,
    ) -> None:
        """Construct an activation quantizer using least squares, ternary."""
        super(ActivationQuantizerLST, self).__init__(
            1, moving_average_mode, moving_average_momentum
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        """Forward pass of quantizing activation using least squares ternary."""
        return super(ActivationQuantizerLST, self).forward(x)

    def _batch_quantization(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return a 2-tuple of (scaling factors, quantized x)."""
        batch_v1, x_q = quantization.quantizer_ls_ternary(x)
        return batch_v1.view(1, -1), x_q

    def _moving_average_quantization(
        self, x: torch.Tensor, vs: List[torch.Tensor]
    ) -> torch.Tensor:
        """Return quantized x using vs."""
        v1 = vs[0]
        _, x_q = quantization.quantizer_ls_ternary(x, v1)
        return x_q


class ActivationQuantizerGF(ActivationQuantizer):
    """Activation greedy foldable quantizer."""

    def __init__(
        self,
        k: int,
        moving_average_mode: str = 'off',
        moving_average_momentum: float = 0.99,
    ) -> None:
        """Construct a greedy-foldable quantizer with `k`-bits."""
        super(ActivationQuantizerGF, self).__init__(
            k, moving_average_mode, moving_average_momentum
        )
        self.k = k

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        """Forward pass of greedy foldable quantizer with `k`-bits."""
        return super(ActivationQuantizerGF, self).forward(x)

    def _batch_quantization(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return a 2-tuple of (scaling factors, quantized x)."""
        batch_vs, x_q = quantization.quantizer_gf(x, self.k)
        return torch.stack(batch_vs), x_q

    def _moving_average_quantization(
        self, x: torch.Tensor, vs: List[torch.Tensor]
    ) -> torch.Tensor:
        """Return quantized x using vs."""
        _, x_q = quantization.quantizer_gf(x, self.k, vs)
        return x_q
