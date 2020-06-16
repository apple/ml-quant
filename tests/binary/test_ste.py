#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

"""Test straight-through estimator."""

import torch

from quant.binary.ste import binarize


def test_ste_sign_forward():
    """Test the forward pass of STESign."""
    x = torch.tensor([42, -42, 42, 42, 0, -1, 1, -4.2, 4.2])
    xb = binarize(x)
    xb_expected = torch.tensor([1, -1, 1, 1, 1, -1, 1, -1, 1])
    assert torch.all(xb.eq(xb_expected))


def test_ste_sign_backward_multiloss():
    """
    Test STESign backward computes gradient correctly.

    x = [x1, x2, ..., xn]
    l = sum(sign(x))
    dl/dxi = 1 iff |xi| <= 1
    """
    x = torch.tensor([42, -42, 0, -1, 1, -0.2, 0.2], requires_grad=True)

    xb = binarize(x)
    loss = xb.sum()
    loss.backward()

    grad_expected = torch.tensor([0, 0, 1, 1, 1, 1, 1])
    assert torch.all(x.grad.eq(grad_expected))
