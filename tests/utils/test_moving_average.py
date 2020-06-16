#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

"""Test moving average."""

import pytest
import torch
import torch.nn as nn

from quant.binary.activation_quantization import ActivationQuantizerLS1
from quant.utils.moving_average import MovingAverage
from quant.binary.quantization import quantizer_ls_1


def test_moving_average():
    """Test moving average."""
    x = torch.tensor([1.0])
    moving_avg = MovingAverage(momentum=torch.tensor([0.9]))
    assert moving_avg(x) == x

    x = torch.tensor([2.0])
    assert torch.allclose(moving_avg(x), torch.tensor([0.9 * 1 + 0.1 * x]))
    prev_result = torch.tensor([0.9 * 1 + 0.1 * x])

    x = torch.tensor([3.0])
    assert torch.allclose(moving_avg(x), torch.tensor([0.9 * prev_result + 0.1 * x]))


def test_moving_average_multiple_momentum():
    """Test moving average with different momentum."""
    x = torch.tensor([2.0, 2.0])
    moving_avg = MovingAverage(momentum=torch.tensor([0.1, 0.2]))
    assert torch.allclose(moving_avg(x), x)

    x = torch.tensor([4.0, 4.0])
    assert torch.allclose(moving_avg(x), torch.tensor([3.8, 3.6]))


def _compute_moving_average_closed_form(i, alpha):
    """Compute the moving average for consecutive positive integers with momentum alpha."""
    return (alpha ** (i + 1) - (i + 1) * alpha + i) / (1 - alpha)


def test_moving_average_train_and_eval():
    """Test moving average with train_and_eval mode set in activation quantizer."""
    alpha = 0.9

    devices = [torch.device('cpu')]
    if torch.cuda.is_available():
        devices.append(torch.device('cuda:0'))

    for device in devices:
        activation_quantizer = ActivationQuantizerLS1('train_and_eval', alpha)
        activation_quantizer.to(device)
        activation_quantizer.train()
        for i in range(10):
            x = i * torch.ones(8, 1, 20, 20, requires_grad=True, device=device)
            x_q = activation_quantizer(x)
            x_q.sum().backward()

            # Moving average internal statistics should be updated
            actual_ma = activation_quantizer.moving_avg_module.moving_average
            ma_i = _compute_moving_average_closed_form(i, alpha)
            expected_ma = torch.tensor(ma_i, device=device).expand_as(actual_ma)
            assert torch.allclose(expected_ma, actual_ma)

            # Quantization should be computed from moving average scalars
            _, expected_quantization = quantizer_ls_1(
                x, torch.tensor([ma_i], device=device).expand(8)
            )
            assert torch.allclose(expected_quantization, x_q)

        activation_quantizer.eval()
        for i in range(5):
            x = i * torch.ones(8, 1, 20, 20, requires_grad=True, device=device)
            activation_quantizer(x).sum().backward()
            actual_ma = activation_quantizer.moving_avg_module.moving_average
            # scalars should be memorized from train and not updated
            expected_ma = torch.tensor(
                _compute_moving_average_closed_form(9, alpha), device=device
            ).expand_as(actual_ma)
            assert torch.allclose(expected_ma, actual_ma)


def test_moving_average_eval_only():
    """Test moving average option with eval_only mode set in activation quantizer."""
    alpha = 0.9

    devices = [torch.device('cpu')]
    if torch.cuda.is_available():
        devices.append(torch.device('cuda:0'))

    for device in devices:
        activation_quantizer = ActivationQuantizerLS1('eval_only', alpha)
        activation_quantizer.to(device)
        activation_quantizer.train()
        for i in range(10):
            x = i * torch.ones(8, 1, 20, 20, requires_grad=True, device=device)
            x_q = activation_quantizer(x)
            x_q.sum().backward()

            # Moving average internal statistics should be updated
            actual_ma = activation_quantizer.moving_avg_module.moving_average
            ma_i = _compute_moving_average_closed_form(i, alpha)
            expected_ma = torch.tensor(ma_i, device=device).expand_as(actual_ma)
            assert torch.allclose(expected_ma, actual_ma)

            # Quantization should NOT be computed from moving average scalars
            assert torch.allclose(x, x_q)

        activation_quantizer.eval()
        for i in range(5):
            x = i * torch.ones(8, 1, 20, 20, requires_grad=True, device=device)
            activation_quantizer(x).sum().backward()
            actual_ma = activation_quantizer.moving_avg_module.moving_average
            # scalars should be memorized from train and not updated
            expected_ma = torch.tensor(
                _compute_moving_average_closed_form(9, alpha), device=device
            ).expand_as(actual_ma)
            assert torch.allclose(expected_ma, actual_ma)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason='requires >= 2 GPUs to run')
def test_moving_average_eval_only_multi_gpu():
    """Test moving average option with eval_only mode set in activation quantizer, with 2 GPUs."""
    alpha = 0.9
    activation_quantizer = ActivationQuantizerLS1('eval_only', alpha)

    activation_quantizer = nn.DataParallel(activation_quantizer, device_ids=[0, 1])
    device = torch.device('cuda:0')
    activation_quantizer.to(device)

    activation_quantizer.train()
    for i in range(10):
        x_gpu0 = i * torch.ones(8, 1, 20, 20, requires_grad=True, device=device)
        x_gpu1 = 42 * torch.ones(8, 1, 20, 20, requires_grad=True, device=device)
        x = torch.cat([x_gpu0, x_gpu1], dim=0)
        x_q = activation_quantizer(x)
        x_q.sum().backward()

        # Moving average internal statistics should be updated
        actual_ma = activation_quantizer.module.moving_avg_module.moving_average
        ma_i = _compute_moving_average_closed_form(i, alpha)
        expected_ma = torch.tensor(ma_i, device=device).expand_as(actual_ma)
        assert torch.allclose(expected_ma, actual_ma)

        # Quantization should NOT be computed from moving average scalars
        assert torch.allclose(x, x_q)

    activation_quantizer.eval()
    for i in range(5):
        x = 42 * torch.ones(16, 1, 20, 20, requires_grad=True, device=device)
        x_q = activation_quantizer(x)
        x_q.sum().backward()
        actual_ma = activation_quantizer.module.moving_avg_module.moving_average

        # scalars should be memorized from train and not updated
        ma_i = _compute_moving_average_closed_form(9, alpha)
        expected_ma = torch.tensor(ma_i, device=device).expand_as(actual_ma)
        assert torch.allclose(expected_ma, actual_ma)

        # Quantization should be using the moving average scalar from the 1st GPU during training
        _, expected = quantizer_ls_1(x, torch.tensor([ma_i], device=device).expand(16))
        assert torch.allclose(x_q, expected)
