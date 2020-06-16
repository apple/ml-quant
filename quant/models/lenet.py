#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

"""
LeNet model.

See http://yann.lecun.com/exdb/lenet/ for more details.
"""

from typing import Callable, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from quant.binary.binary_conv import QuantConv2d


class QLeNet5(nn.Module):
    """LeNet-5 model."""

    def __init__(
        self,
        loss_fn: Callable[..., torch.Tensor],
        conv1_filters: int = 20,
        conv2_filters: int = 50,
        output_classes: int = 10,
        x_quant: str = 'fp',
        w_quant: str = 'fp',
        clamp: Optional[Dict] = None,
        moving_average_mode: str = 'off',
        moving_average_momentum: float = 0.99,
    ) -> None:
        """
        Initialize weights and biases for LeNet model.

        Args:
            loss_fn: loss function of the model
            conv1_filters: number of convolutional feature maps of the first conv layer
            conv2_filters: number of convolutional feature maps of the second conv layer
            output_classes: number of output classes
            x_quant: quantization scheme for activations,
                see :mod:`~quant.binary.binary_conv`.
            w_quant: quantization scheme for weights,
                see :mod:`~quant.binary.binary_conv`.
            clamp: clamping scheme for activations.
                It should have a key named "kind" indicating the kind of clamping function
                and other keys indicating other potential arguments.
                See :mod:`~quant.binary.binary_conv`.
            moving_average_mode: moving average mode to use
                see :class:`~quant.binary.activation_quantization.ActivationQuantizer`.
            moving_average_momentum: momentum for moving average
                update, see :class:`~quant.binary.activation_quantization.ActivationQuantizer`.
        """
        super(QLeNet5, self).__init__()
        # loss_fn is a loss function in torch.nn.functional
        setattr(self, 'loss_fn', loss_fn)

        self.conv1_filters = conv1_filters
        self.conv2_filters = conv2_filters
        self.output_classes = output_classes
        self.x_quant = x_quant
        self.w_quant = w_quant

        self.conv1 = nn.Conv2d(1, conv1_filters, 5, stride=1)
        self.bn_conv1 = nn.BatchNorm2d(conv1_filters, eps=1e-4, momentum=0.1, affine=False)
        self.conv2 = QuantConv2d(
            x_quant, w_quant, conv1_filters, conv2_filters, 5,
            clamp, moving_average_mode, moving_average_momentum, stride=1
        )

        self.bn_conv2 = nn.BatchNorm2d(conv1_filters, eps=1e-4, momentum=0.1, affine=False)
        self.fc1 = nn.Linear(conv2_filters * 4 * 4, conv2_filters * output_classes)
        self.fc2 = nn.Linear(conv2_filters * output_classes, output_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        """Forward pass of LeNet5 model."""
        # first layer full precision
        x = self.conv1(x)
        x = self.bn_conv1(F.relu(x, inplace=True))
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        x = F.relu(self.conv2(self.bn_conv2(x)), inplace=True)
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        x = x.view(-1, self.conv2_filters * 4 * 4)
        x = self.fc1(x)
        x = F.relu(x, inplace=True)

        # last layer full precision
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
