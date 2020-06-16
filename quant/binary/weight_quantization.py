#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

"""Weight quantization."""

import torch
import torch.nn as nn

import quant.binary.quantization as quantization


class WeightQuantizerLS1(nn.Module):
    """
    Weight quantizer using least squares, 1 bit.

    In training mode, the optimal scalars are computed and cached.
    In eval mode, the cached scalars are used to compute the quantization.
    """

    def __init__(self, size: int) -> None:
        """Construct a weight quantizer using least squares with 1 bit."""
        super(WeightQuantizerLS1, self).__init__()
        self.register_buffer('v1', torch.tensor([0.0] * size))

    def forward(self, w: torch.Tensor) -> torch.Tensor:  # type: ignore
        """Forward pass of quantizing weight using least squares 1 bit."""
        if self.training:
            v1, w_q = quantization.quantizer_ls_1(w)
            self.v1.copy_(v1)  # type: ignore
        else:
            _, w_q = quantization.quantizer_ls_1(w, self.v1)  # type: ignore
        return w_q


class WeightQuantizerLS2(nn.Module):
    """
    Weight quantizer using least squares, 2 bits.

    In training mode, the optimal scalars are computed and cached.
    In eval mode, the cached scalars are used to compute the quantization.
    """

    def __init__(self, size: int) -> None:
        """Construct a weight quantizer using least squares with 2 bits."""
        super(WeightQuantizerLS2, self).__init__()
        self.register_buffer('v1', torch.tensor([0.0] * size))
        self.register_buffer('v2', torch.tensor([0.0] * size))

    def forward(self, w: torch.Tensor, skip: int = 3) -> torch.Tensor:  # type: ignore
        """Forward pass of quantizing weight using least squares 2 bits."""
        if self.training:
            v1, v2, w_q = quantization.quantizer_ls_2(w, skip=skip)
            self.v1.copy_(v1)  # type: ignore
            self.v2.copy_(v2)  # type: ignore
        else:
            _, _, w_q = quantization.quantizer_ls_2(w, self.v1, self.v2, skip=skip)  # type: ignore
        return w_q


class WeightQuantizerLST(nn.Module):
    """
    Weight quantizer using least squares, ternary.

    In training mode, the optimal scalars are computed and cached.
    In eval mode, the cached scalars are used to compute the quantization.
    """

    def __init__(self, size: int) -> None:
        """Construct a weight quantizer using least squares ternary."""
        super(WeightQuantizerLST, self).__init__()
        self.register_buffer('v1', torch.tensor([0.0] * size))

    def forward(self, w: torch.Tensor, skip: int = 3) -> torch.Tensor:  # type: ignore
        """Forward pass of quantizing weight using least squares ternary."""
        if self.training:
            v1, w_q = quantization.quantizer_ls_ternary(w, skip=skip)
            self.v1.copy_(v1)  # type: ignore
        else:
            _, w_q = quantization.quantizer_ls_ternary(w, self.v1, skip=skip)  # type: ignore
        return w_q


class WeightQuantizerGF(nn.Module):
    """
    Weight greedy foldable quantizer.

    In training mode, the optimal scalars are computed and cached.
    In eval mode, the cached scalars are used to compute the quantization.
    """

    def __init__(self, size: int, k: int) -> None:
        """Construct a greedy-foldable quantizer with `k`-bits."""
        super(WeightQuantizerGF, self).__init__()
        self.k = k
        for i in range(1, k + 1):
            self.register_buffer(f'v{i}', torch.tensor([0.0] * size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        """Forward pass of greedy foldable quantizer with `k`-bits."""
        if self.training:
            vs, x_q = quantization.quantizer_gf(x, k=self.k)
            for i in range(self.k):
                getattr(self, f'v{i+1}').copy_(vs[i])
        else:
            vs = [getattr(self, f'v{i+1}') for i in range(self.k)]
            _, x_q = quantization.quantizer_gf(x, k=self.k, vs=vs)
        return x_q
