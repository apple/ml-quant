#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

"""Test binary convolution."""

import itertools

import pytest
import torch
import torch.nn as nn


from quant.binary.binary_conv import QuantConv2d


def test_fp_quant_conv2d_eq_nn_conv2d():
    """Test full precision QuantConv2d equals to regular Conv2d."""
    torch.manual_seed(1234)
    x = torch.randn(8, 3, 100, 100, requires_grad=True)
    x_copy = x.clone().detach().requires_grad_(True)

    nn_conv2d = nn.Conv2d(3, 30, 5)
    expected_out = nn_conv2d(x)
    expected_loss = expected_out.sum()

    scaled_conv2d = QuantConv2d('fp', 'fp', 3, 30, 5)
    scaled_conv2d.weight = nn.Parameter(nn_conv2d.weight, requires_grad=True)
    scaled_conv2d.bias = nn.Parameter(nn_conv2d.bias, requires_grad=True)
    actual_out = scaled_conv2d(x_copy)
    actual_loss = actual_out.sum()

    expected_loss.backward()
    actual_loss.backward()

    assert torch.all(expected_out.eq(actual_out))
    assert torch.all(x.grad.eq(x_copy.grad))


def test_ls1_quant_conv2d_sanity():
    """Sanity check for least squares 1-bit scaled binary Conv2d."""
    torch.manual_seed(1234)
    x = torch.randn(4, 3, 8, 8)
    conv2d = QuantConv2d('ls-1', 'ls-1', 3, 16, (2, 2))
    y = conv2d(x)
    # the absolute value of each element in the input x and weight should be at most 1
    # this is a quick sanity check for each of the 16 filters
    for i in range(4):
        for j in range(16):
            assert torch.max(y[i, j].abs()) <= 2 * 2 * 3 + conv2d.bias[j]


def test_w_ls1_x_fp_quant_conv2d():
    """Basic test for ls-1 weight, fp activation (input)."""
    x = torch.zeros(1, 3, 8, 8)
    x.data[0, :, :4, 4:] = -1
    x.data[0, :, 4:, :4] = 2
    x.data[0, :, 4:, 4:] = -3
    conv2d = QuantConv2d(
        'fp', 'ls-1', 3, 1, (4, 4), stride=4, bias=False
    )
    y = conv2d(x).squeeze()
    assert y.shape == (2, 2)
    assert y[0, 0] == 0
    assert torch.isclose(y[1, 0], -2 * y[0, 1])
    assert torch.isclose(y[1, 1], 3 * y[0, 1])


def test_quant_conv2d_parameter_group_keys():
    """Test parameter groups are separated correctly."""
    clamp = {'alpha': 2, 'kind': 'symmetric'}
    conv2d = QuantConv2d(
        'ls-2', 'ls-1', 3, 1, (4, 4), clamp=clamp, stride=4, bias=False
    )
    assert len(conv2d.quantized_parameters['fp']) == 0
    assert len(conv2d.quantized_parameters['ls-1']) == 1
    assert set(conv2d.quantized_parameters.keys()) - {'fp', 'ls-1'} == set()
    assert len(list(conv2d.parameters())) == 1

    conv2d = QuantConv2d('ls-2', 'ls-2', 3, 1, (4, 4), clamp=clamp, stride=4)
    assert len(conv2d.quantized_parameters['fp']) == 1
    assert len(conv2d.quantized_parameters['ls-2']) == 1
    assert set(conv2d.quantized_parameters.keys()) - {'fp', 'ls-2'} == set()
    assert len(list(conv2d.parameters())) == 2


def test_quant_conv2d_combinations():
    """Test different combinations of configurations to see they can be created."""
    schemes = ['fp', 'ls-1', 'ls-2', 'ls-T', 'gf-2', 'gf-3']
    for x_scheme, w_scheme in itertools.product(schemes, schemes):
        QuantConv2d(x_scheme, w_scheme, 3, 1, (4, 4))

    with pytest.raises(ValueError):
        QuantConv2d('ls', 'ls-1', 3, 1, (4, 4))

    with pytest.raises(ValueError):
        QuantConv2d('l2', 'ls-1', 3, 1, (4, 4))

    with pytest.raises(ValueError):
        QuantConv2d('ls-1', 'ls-3', 3, 1, (4, 4))

    with pytest.raises(ValueError):
        QuantConv2d('ls-1', 'l2', 3, 1, (4, 4))

    with pytest.raises(ValueError):
        QuantConv2d('ls-1', 'ls-2', 3, 1, (4, 4), clamp={'kind': 'sym'})
