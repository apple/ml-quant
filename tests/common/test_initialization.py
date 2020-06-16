#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

"""Test initialization."""

from unittest.mock import patch

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, SGD
import torch.optim.lr_scheduler as lr_scheduler

from quant.common.initialization import _get_best_gpus, get_model, get_optimizer, get_lr_scheduler
from quant.models.lenet import QLeNet5
from quant.utils.linear_lr_scheduler import LinearLR


def test_get_model_cpu():
    """Test get model factory on CPU."""
    arch = {'conv1_filters': 20, 'conv2_filters': 50, 'output_classes': 10}
    model = get_model('lenet5', F.nll_loss, arch, torch.device('cpu'), 0)

    assert isinstance(model, QLeNet5)
    assert model.loss_fn == F.nll_loss
    assert next(model.parameters()).device.type == 'cpu'


def test_get_model_single_gpu():
    """Test get model factory on single GPU."""
    if not torch.cuda.is_available():
        return

    arch = {'conv1_filters': 20, 'conv2_filters': 50, 'output_classes': 10}
    model = get_model('lenet5', F.nll_loss, arch, torch.device('cuda:0'), 1)

    assert isinstance(model, QLeNet5)
    assert model.loss_fn == F.nll_loss
    assert next(model.parameters()).device.type == 'cuda'


def test_get_model_multi_gpu():
    """Test get model factory on single GPU."""
    if torch.cuda.device_count() <= 1:
        return

    arch = {'conv1_filters': 20, 'conv2_filters': 50, 'output_classes': 10}
    model = get_model('lenet5', F.nll_loss, arch, torch.device('cuda:0'), 2)

    assert isinstance(model, nn.DataParallel)
    assert model.module.loss_fn == F.nll_loss


@patch('torch.cuda.device_count')
@patch('torch.cuda.get_device_capability')
def test_get_best_gpus(capability_mock, device_count_mock):
    """Test _get_best_gpus returns the best GPUs."""
    def device_capability_side_effect(device_id):
        if device_id == 0:
            return 6, 0
        if device_id == 1:
            return 7, 5
        if device_id == 2:
            return 6, 5

    assert torch.cuda.device_count is device_count_mock
    assert torch.cuda.get_device_capability is capability_mock
    device_count_mock.return_value = 3
    capability_mock.side_effect = device_capability_side_effect

    device_ids = _get_best_gpus(2)

    assert set(device_ids) == {1, 2}


def test_get_optimizer():
    """Test get optimizer factory."""
    model = QLeNet5(F.nll_loss)

    optimizer = get_optimizer(model.parameters(), {'algorithm': 'sgd', 'lr': 0.1})
    assert isinstance(optimizer, SGD)

    optimizer = get_optimizer(model.parameters(), {'algorithm': 'adam', 'lr': 0.1})
    assert isinstance(optimizer, Adam)


def test_get_linear_lr_scheduler():
    """Test get linear lr scheduler."""
    model = QLeNet5(F.nll_loss)
    optimizer = get_optimizer(model.parameters(), {'algorithm': 'sgd', 'lr': 0.1})

    scheduler = get_lr_scheduler(
        optimizer,
        {'scheduler': 'linear_lr', 'min_lr': 1e-5}, 80, 100
    )

    assert isinstance(scheduler, LinearLR)
    # This test just check we can construct a LinearLR,
    # test_linear_lr_scheduler actually tests its behavior


def test_get_step_lr_scheduler():
    """Test get step lr scheduler."""
    model = QLeNet5(F.nll_loss)
    optimizer = get_optimizer(model.parameters(), {'algorithm': 'sgd', 'lr': 0.1})

    scheduler = get_lr_scheduler(
        optimizer,
        {'scheduler': 'step_lr', 'step_size': 1, 'gamma': 0.7}, 5, 100
    )

    assert isinstance(scheduler, lr_scheduler.StepLR)
    for _ in range(100):
        assert optimizer.param_groups[0]['lr'] == 0.1
        optimizer.step()
        scheduler.step()

    assert optimizer.param_groups[0]['lr'] == 0.7 * 0.1


def test_get_multi_step_lr_scheduler():
    """Test get multi step lr scheduler."""
    model = QLeNet5(F.nll_loss)
    optimizer = get_optimizer(model.parameters(), {'algorithm': 'sgd', 'lr': 0.1})
    scheduler = get_lr_scheduler(
        optimizer,
        {'scheduler': 'multi_step_lr', 'milestones': [30, 70], 'gamma': 0.7}, 70, 100
    )

    assert isinstance(scheduler, lr_scheduler.MultiStepLR)
    for _ in range(30 * 100):
        assert optimizer.param_groups[0]['lr'] == 0.1
        optimizer.step()
        scheduler.step()

    for _ in range(40 * 100):
        assert optimizer.param_groups[0]['lr'] == 0.7 * 0.1
        optimizer.step()
        scheduler.step()

    assert optimizer.param_groups[0]['lr'] == 0.7 * 0.7 * 0.1


def test_get_lambda_lr_scheduler():
    """Test get lambda lr scheduler."""
    model = QLeNet5(F.nll_loss)
    optimizer = get_optimizer(model.parameters(), {'algorithm': 'sgd', 'lr': 0.1})

    lr_lambda = """lambda s: next(
        v for (a, b), v in {(0, 200): 1, (200, 1000): 0.75}.items() if a <= s < b
    )"""
    scheduler = get_lr_scheduler(
        optimizer,
        {'scheduler': 'lambda_lr', 'lr_lambda': lr_lambda}, 10, 100
    )

    assert isinstance(scheduler, lr_scheduler.LambdaLR)
    for _ in range(200):
        assert optimizer.param_groups[0]['lr'] == 0.1
        optimizer.step()
        scheduler.step()

    assert optimizer.param_groups[0]['lr'] == 0.75 * 0.1
