#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

"""Test ResNet."""

import torch
import torch.nn.functional as F

from quant.models.resnet import QResNet


REGULAR_BASIC_CONFIG = {
    "block": "regular",
    "layer0": {
        "bias": False,
        "kernel_size": 7,
        "maxpool": {"kernel_size": 3, "padding": 1, "stride": 2, "type": "maxpool2d"},
        "n_in_channels": 64,
        "padding": 3,
        "stride": 2,
    },
    "layer1": {"clamp": {"kind": "identity"}, "w_quant": "fp", "x_quant": "fp"},
    "layer2": {"clamp": {"kind": "identity"}, "w_quant": "fp", "x_quant": "fp"},
    "layer3": {"clamp": {"kind": "identity"}, "w_quant": "fp", "x_quant": "fp"},
    "layer4": {"clamp": {"kind": "identity"}, "w_quant": "fp", "x_quant": "fp"},
    "nonlins": ["relu", "relu"],
    "num_blocks": [2, 2, 2, 2],
    "output_classes": 1000,
}

XNOR_BASIC_CONFIG = {
    "block": "xnor",
    "layer0": {
        "bias": False,
        "kernel_size": 7,
        "maxpool": {"kernel_size": 3, "padding": 1, "stride": 2, "type": "maxpool2d"},
        "n_in_channels": 64,
        "padding": 3,
        "stride": 2,
    },
    "layer1": {
        "clamp": {"alpha": 2, "kind": "symmetric"},
        "double_shortcut": False,
        "w_quant": "ls-1",
        "x_quant": "ls-1",
    },
    "layer2": {
        "clamp": {"alpha": 2, "kind": "symmetric"},
        "double_shortcut": False,
        "w_quant": "ls-1",
        "x_quant": "ls-1",
    },
    "layer3": {
        "clamp": {"alpha": 2, "kind": "symmetric"},
        "double_shortcut": False,
        "w_quant": "ls-1",
        "x_quant": "ls-1",
    },
    "layer4": {
        "clamp": {"alpha": 2, "kind": "symmetric"},
        "double_shortcut": False,
        "w_quant": "ls-1",
        "x_quant": "ls-1",
    },
    "nonlins": ["prelu", "prelu"],
    "num_blocks": [2, 2, 2, 2],
    "output_classes": 1000,
}

XNOR_BASIC_DOUBLE_SC_CONFIG = {
    "block": "xnor",
    "layer0": {
        "bias": False,
        "kernel_size": 7,
        "maxpool": {"kernel_size": 3, "padding": 1, "stride": 2, "type": "maxpool2d"},
        "n_in_channels": 64,
        "padding": 3,
        "stride": 2,
    },
    "layer1": {
        "clamp": {"alpha": 2, "kind": "symmetric"},
        "double_shortcut": True,
        "w_quant": "ls-1",
        "x_quant": "ls-1",
    },
    "layer2": {
        "clamp": {"alpha": 2, "kind": "symmetric"},
        "double_shortcut": True,
        "w_quant": "ls-1",
        "x_quant": "ls-1",
    },
    "layer3": {
        "clamp": {"alpha": 2, "kind": "symmetric"},
        "double_shortcut": True,
        "w_quant": "ls-1",
        "x_quant": "ls-1",
    },
    "layer4": {
        "clamp": {"alpha": 2, "kind": "symmetric"},
        "double_shortcut": True,
        "w_quant": "ls-1",
        "x_quant": "ls-1",
    },
    "nonlins": ["prelu", "prelu"],
    "num_blocks": [2, 2, 2, 2],
    "output_classes": 1000,
}


def test_regular_basic_block_forward():
    """Test forward pass of regular basic block."""
    torch.manual_seed(1234)
    x = torch.randn(4, 3, 32, 32)
    resnet = QResNet(loss_fn=F.cross_entropy, **REGULAR_BASIC_CONFIG)
    y = resnet(x)
    assert y.shape == (4, 1000)


def test_xnor_basic_block_forward():
    """Test forward pass of xnor basic block."""
    torch.manual_seed(1234)
    x = torch.randn(4, 3, 32, 32)
    resnet = QResNet(loss_fn=F.cross_entropy, **XNOR_BASIC_CONFIG)
    y = resnet(x)
    assert y.shape == (4, 1000)


def test_xnor_basic_with_double_shortcut_forward():
    """Test forward pass of xnor basic block with double shortcut."""
    torch.manual_seed(1234)
    x = torch.randn(4, 3, 32, 32)
    resnet = QResNet(loss_fn=F.cross_entropy, **XNOR_BASIC_DOUBLE_SC_CONFIG)
    y = resnet(x)
    assert y.shape == (4, 1000)
