#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

"""Test linear learning rate scheduler."""

import math

import pytest
import torch.nn as nn
import torch.optim as optim

from quant.utils.linear_lr_scheduler import LinearLR


def test_linear_lr_scheduler():
    """Test linear lr scheduler."""
    model = nn.Conv2d(3, 32, (2, 2), bias=False)
    optimizer = optim.Adam(model.parameters(), lr=0.0002)
    epochs = 120
    total_examples = 1281167
    batch_size = 256
    steps_per_epoch = int(math.ceil(total_examples / batch_size))
    scheduler = LinearLR(optimizer, 2e-7, epochs, steps_per_epoch)

    lrs = []
    for epoch in range(epochs):
        for batch in range(steps_per_epoch):
            lrs.append(optimizer.param_groups[0]['lr'])
            optimizer.step()
            scheduler.step()

    assert lrs[0] == 0.0002
    assert pytest.approx(lrs[1], 0.000199999663866, 1e-14)
    assert pytest.approx(lrs[2], 0.000199999327731, 1e-14)
    assert pytest.approx(lrs[80], 0.000199973109244, 1e-14)
    assert pytest.approx(lrs[160], 0.000199946218487, 1e-14)
    assert pytest.approx(lrs[60000], 0.000179831932773, 1e-14)
    assert lrs[epochs * steps_per_epoch - 1] == 2e-7
