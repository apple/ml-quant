#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

"""Straight-through estimator."""

from typing import Any, NewType

import torch
from torch.autograd import Function

BinaryTensor = NewType('BinaryTensor', torch.Tensor)  # A type where each element is in {-1, 1}


def binary_sign(x: torch.Tensor) -> BinaryTensor:
    """Return -1 if x < 0, 1 if x >= 0."""
    return x.sign() + (x == 0).type(torch.float)  # type: ignore


class STESign(Function):
    """
    Binarize tensor using sign function.

    Straight-Through Estimator (STE) is used to approximate the gradient of sign function.

    See:
    Bengio, Yoshua, Nicholas Léonard, and Aaron Courville.
    "Estimating or propagating gradients through stochastic neurons for
     conditional computation." arXiv preprint arXiv:1308.3432 (2013).
    """

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor) -> BinaryTensor:  # type: ignore
        """
        Return a Sign tensor.

        Args:
            ctx: context
            x: input tensor

        Returns:
            Sign(x) = (x>=0) - (x<0)
            Output type is float tensor where each element is either -1 or 1.
        """
        ctx.save_for_backward(x)
        sign_x = binary_sign(x)
        return sign_x

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:  # type: ignore  # pragma: no cover (since this is called by C++ code) # noqa: E501
        """
        Compute gradient using STE.

        Args:
            ctx: context
            grad_output: gradient w.r.t. output of Sign

        Returns:
            Gradient w.r.t. input of the Sign function
        """
        x, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[x.gt(1)] = 0
        grad_input[x.lt(-1)] = 0
        return grad_input


# Convenience function to binarize tensors
binarize = STESign.apply    # type: ignore
