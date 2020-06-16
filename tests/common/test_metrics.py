#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

"""Test metrics."""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from quant.common.metrics import LossMetric, Top1Accuracy, TopKAccuracy


def test_loss_metric_no_accumulate():
    """Test loss metric returns correct value with no accumulate."""
    criterion = F.nll_loss
    metric = LossMetric(criterion, accumulate=False)
    model = nn.LogSoftmax(dim=1)
    X = torch.randn(3, 5)
    output = model(X)
    target = torch.tensor([1, 0, 4])
    metric.update(output, target)

    assert F.nll_loss(output, target).item() == metric.compute()

    # Check this is true after re-computation
    assert F.nll_loss(output, target).item() == metric.compute()

    # Check update
    Y = torch.randn(3, 5)
    output2 = model(Y)
    metric.update(output2, target)
    assert F.nll_loss(output2, target).item() == metric.compute()

    # Check this is true after reset & re-computation
    metric.reset()
    metric.update(output, target)
    assert F.nll_loss(output, target).item() == metric.compute()


def test_loss_metric_accumulate():
    """Test loss metric returns correct value with accumulate."""
    criterion = F.nll_loss
    metric = LossMetric(criterion, accumulate=True)
    model = nn.LogSoftmax(dim=1)
    X = torch.randn(3, 5)
    output = model(X)
    target = torch.tensor([1, 0, 4])
    metric.update(output, target)

    assert F.nll_loss(output, target).item() == pytest.approx(metric.compute())

    # Check this is true after re-computation
    assert F.nll_loss(output, target).item() == pytest.approx(metric.compute())

    # Check update
    Y = torch.randn(3, 5)
    output2 = model(Y)
    metric.update(output2, target)
    assert F.nll_loss(torch.cat([output, output2]), torch.cat([target, target])).item() \
        == pytest.approx(metric.compute())

    # Check this is true after reset & re-computation
    metric.reset()
    metric.update(output, target)
    assert F.nll_loss(output, target).item() == pytest.approx(metric.compute())


def test_top_1_accuracy_metric_no_accumulate():
    """Test top-1 accuracy metric returns correct value with no accumulate."""
    metric = Top1Accuracy(accumulate=False)

    metric.update(torch.tensor([[0.1, 0.2, 0.3]]), torch.tensor([2]))
    assert metric.compute() == 1.0

    # Check this is true after re-computation
    assert metric.compute() == 1.0

    metric.update(torch.tensor([[0.1, 0.2, 0.3]]), torch.tensor([1]))
    assert metric.compute() == 0

    # Check after reset & re-computation
    metric.reset()
    metric.update(torch.tensor([[0.1, 0.2, 0.3]]), torch.tensor([2]))
    assert metric.compute() == 1.0


def test_top_1_accuracy_metric_accumulate():
    """Test top-1 accuracy metric returns correct value with accumulate."""
    metric = Top1Accuracy(accumulate=True)

    metric.update(torch.tensor([[0.1, 0.2, 0.3]]), torch.tensor([2]))
    assert metric.compute() == 1.0

    metric.update(torch.tensor([[0.1, 0.2, 0.3]]), torch.tensor([1]))
    assert metric.compute() == 0.5

    metric.update(torch.tensor([[0.1, 0.2, 0.3]]), torch.tensor([0]))
    assert metric.compute() == 1 / 3

    # Check this is true after re-computation
    assert metric.compute() == 1 / 3

    # Check this is true after reset & re-computation
    metric.reset()
    metric.update(torch.tensor([[0.1, 0.2, 0.3]]), torch.tensor([2]))
    assert metric.compute() == 1.0


def test_top_k_accuracy_metric_no_accumulate():
    """Test top-k accuracy metric returns correct value with no accumulate."""
    output = torch.tensor([[0.1, 0.2, 0.3, 0, 0.5],
                           [0.2, 0.3, 0.4, 0.1, 0]])

    metric_k2 = TopKAccuracy(2, accumulate=False)
    metric_k2.update(output, torch.tensor([4, 0]))
    assert metric_k2.compute() == 0.5

    metric_k2.update(torch.tensor([[0.1, 0.5, 0.3, 0.2, 0.4]]), torch.tensor([1]))
    assert metric_k2.compute() == 1.0

    # Check re-computation does not change value
    assert metric_k2.compute() == 1.0

    # Check reset works
    metric_k2.reset()
    metric_k2.update(output, torch.tensor([4, 0]))
    assert metric_k2.compute() == 0.5


def test_top_k_accuracy_metric_accumulate():
    """Test top-k accuracy metric returns correct value with accumulate."""
    output = torch.tensor([[0.1, 0.2, 0.3, 0, 0.5],
                           [0.2, 0.3, 0.4, 0.1, 0]])

    metric_k2 = TopKAccuracy(2, accumulate=True)
    metric_k2.update(output, torch.tensor([4, 0]))
    assert metric_k2.compute() == 0.5

    metric_k3 = TopKAccuracy(3, accumulate=True)
    metric_k3.update(output, torch.tensor([4, 0]))
    assert metric_k3.compute() == 1.0

    metric_k2.update(torch.tensor([[0.1, 0.5, 0.3, 0.2, 0.4]]), torch.tensor([1]))
    assert metric_k2.compute() == 2 / 3

    # Check re-computation does not change value
    assert metric_k2.compute() == 2 / 3

    # Check reset works
    metric_k2.reset()
    metric_k2.update(output, torch.tensor([4, 0]))
    assert metric_k2.compute() == 0.5
