#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

"""Test training and testing loop."""

import unittest.mock as mock

import pytest
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from quant.common.initialization import get_optimizer, get_lr_scheduler
from quant.common.metrics import LossMetric
from quant.common.training import train, evaluate
from quant.models.lenet import QLeNet5

from tests.data.helpers import RandomDataset


@pytest.fixture
def random_data_loader():
    torch.manual_seed(260)
    loader = DataLoader(RandomDataset(2), batch_size=32, num_workers=4, shuffle=False)
    return loader


def test_training_loop(random_data_loader):
    """Test the training loop."""
    device = torch.device('cpu')
    model = QLeNet5(F.nll_loss).to(device)
    metrics = {
        'Loss': LossMetric(model.loss_fn, accumulate=False)
    }
    optimizer = get_optimizer(model.parameters(), {'algorithm': 'sgd', 'lr': 0.1})
    scheduler = get_lr_scheduler(
        optimizer,
        {'scheduler': 'step_lr', 'step_size': 1, 'gamma': 0.7},
        3,
        len(random_data_loader)
    )
    fake_hook = mock.MagicMock()
    hooks = [fake_hook]

    losses = []
    for epoch in range(1, 3):
        train(
            model=model, train_loader=random_data_loader, metrics=metrics,
            optimizer=optimizer, scheduler=scheduler, device=device, epoch=epoch,
            log_interval=4, hooks=hooks
        )
        losses.append(metrics['Loss'].compute())

    # Ensure that hooks are called and loss is changing
    assert fake_hook.called
    assert losses[1] != losses[0]


def test_test_loop(random_data_loader):
    """Test the test loop."""
    device = torch.device('cpu')
    model = QLeNet5(F.nll_loss).to(device)
    metrics = {
        'Loss': LossMetric(model.loss_fn, accumulate=False)
    }
    fake_hook = mock.MagicMock()
    hooks = [fake_hook]
    evaluate(model=model, test_loader=random_data_loader, metrics=metrics, device=device,
             epoch=1, hooks=hooks)

    # Ensure that hooks are called and metric has value
    assert fake_hook.called
    assert isinstance(metrics['Loss'].compute(), float)
