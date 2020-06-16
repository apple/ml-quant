#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

"""Quantization functions and classes."""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from quant.binary.optimal import opt_v1
from quant.binary.ste import binarize, binary_sign


def clamp_identity(x: torch.Tensor) -> torch.Tensor:
    """Identity clamp."""
    return x


def clamp_symmetric(x: torch.Tensor, alpha: float) -> torch.Tensor:
    """Clamp x to [-alpha, +alpha]."""
    return x.clamp(-alpha, alpha)


class QuantizerFP(nn.Module):
    """Weight / activation quantizer using full precision."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        """Forward pass of full-precision quantizer."""
        return x


def quantizer_ls_1(
    x: torch.Tensor, v1: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Return (scaling factors, 1-bit optimal least-squares scaled binary quantization).

    If v1 is provided, it is directly used to compute the quantization.
    If v1 is not provided, it is computed as well.

    Reference:
    Rastegari, Mohammad, et al.
    "Xnor-net: Imagenet classification using binary convolutional neural networks."
    European conference on computer vision. Springer, Cham, 2016.

    Args:
        x: A 4D tensor
        v1: A vector of scaling factors
    """
    x_data = x.clone().detach()
    if v1 is None:
        v1 = x_data.abs().mean(dim=-1).mean(dim=-1).mean(dim=-1)
    return v1, v1.view(-1, 1, 1, 1) * binarize(x)


def quantizer_ls_2(
    x: torch.Tensor,
    v1: Optional[torch.Tensor] = None,
    v2: Optional[torch.Tensor] = None,
    skip: int = 3,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Return (v1, v2, 2-bits optimal least-squares scaled binary quantization).

    If v1 is provided, it is directly used to compute v2 and the quantization.
    If v1 is not provided, it is computed as well.

    Args:
        x: A 4D tensor
        v1: A vector of scaling factors, v1
        v2: A vector of scaling factors, v2
        skip: increment in potential solution space to speed up computation
    """
    x_data = x.view(x.shape[0], -1).clone().detach()
    if v1 is None:
        v1 = opt_v1(x_data, ternary=False, skip=skip)
    else:
        v1 = v1.view(-1, 1)

    if v2 is None:
        residual = x_data - v1 * binary_sign(x_data)
        v2 = residual.abs().mean(dim=-1, keepdim=True)
    else:
        v2 = v2.view(-1, 1)

    v1_reshaped = v1.view(x.shape[0], 1, 1, 1)
    b1 = binarize(x)
    return v1.view(-1), v2.view(-1), \
        v1_reshaped * b1 + v2.view(x.shape[0], 1, 1, 1) * binarize(x - v1_reshaped * b1)


def quantizer_ls_ternary(
    x: torch.Tensor, v1: Optional[torch.Tensor] = None, skip: int = 3
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Return (v1 scaling factors, optimal ternary least-squares scaled binary quantization).

    If v1 is provided, it is directly used to compute the quantization (v2 = v1).
    If v1 is not provided, it is computed as well.

    Args:
        x: A 4D tensor
        v1: A vector of scaling factors, v1
        skip: increment in potential solution space to speed up computation
    """
    x_data = x.view(x.shape[0], -1).clone().detach()
    if v1 is None:
        v1 = opt_v1(x_data, ternary=True, skip=skip)

    v1_reshaped = v1.view(x.shape[0], 1, 1, 1)
    b1 = binarize(x)
    return v1.view(-1), v1_reshaped * (b1 + binarize(x - v1_reshaped * b1))


def quantizer_gf(
    x: torch.Tensor, k: int, vs: Optional[List[torch.Tensor]] = None
) -> Tuple[List[torch.Tensor], torch.Tensor]:
    """
    Return (List of greedy v_is, greedy foldable quantization with k-bits).

    Args:
        x: A 4D tensor
        k: Number of bits
        vs: Scaling factors v_1 to v_k.
            The tensor at position i in the list represents the tensor representing v_{i+1}.
    """
    if vs is not None:
        if len(vs) != k:  # pragma: no cover
            raise ValueError(
                'If vs is passed in, all vs from v_1 to v_k must be passed in (could be None).'
            )

    residual = x.view(x.shape[0], -1).clone().detach()
    result = 0
    saved_vs = []
    for i in range(k):
        if vs is not None:
            v = vs[i]
        else:
            v = residual.abs().mean(dim=-1)
        saved_vs.append(v)
        residual = residual - v.view(-1, 1) * binary_sign(residual)
        result = result + v.view(-1, 1, 1, 1) * binarize(x - result)

    return saved_vs, result  # type: ignore
