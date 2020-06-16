#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

"""
ResNet model.

See `Deep Residual Learning for Image Recognition`_ for more details.

.. _Deep Residual Learning for Image Recognition: https://arxiv.org/abs/1512.03385
"""

from typing import Callable, Dict, List, Optional, Union

import torch
import torch.nn as nn

from quant.binary.binary_conv import QuantConv2d

non_linearity_map = {
    'relu': nn.ReLU,
    'prelu': nn.PReLU,
    'identity': nn.Identity,
}


class RegularBasicBlock(nn.Module):
    """ResNet regular basic block."""

    def __init__(
        self, in_planes: int, planes: int, x_quant: str, w_quant: str,
        nonlins: List[str], stride: int = 1,
        clamp: Optional[Dict] = None,
        moving_average_mode: str = 'off',
        moving_average_momentum: float = 0.99,
    ) -> None:
        """
        Build ResNet regular basic block.

        Args:
            in_planes: the number of in-channels for the block
            planes: the number of out-channels for the block
            x_quant: quantization scheme for activations,
                see :mod:`~quant.binary.binary_conv`.
            w_quant: quantization scheme for weights,
                see :mod:`~quant.binary.binary_conv`.
            nonlins: non-linearities for the black. It should be a list of two
                strings, where each string is in {'relu', 'prelu', 'identity'}.
            stride: stride size
            clamp: clamping scheme for activations.
                It should have a key named "kind" indicating the kind of clamping function
                and other keys indicating other potential arguments.
                See :mod:`~quant.binary.binary_conv`.
            moving_average_mode: moving average mode to use,
                see :class:`~quant.binary.activation_quantization.ActivationQuantizer`.
            moving_average_momentum: momentum for moving average update,
                see :class:`~quant.binary.activation_quantization.ActivationQuantizer`.
        """
        super(RegularBasicBlock, self).__init__()
        if len(nonlins) != 2:
            raise ValueError('There should be 2 non-linearities.')

        self.conv1 = QuantConv2d(
            x_quant, w_quant, in_planes, planes, 3, clamp,
            moving_average_mode, moving_average_momentum, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.nonlin1 = non_linearity_map[nonlins[0]]()

        self.conv2 = QuantConv2d(
            x_quant, w_quant, planes, planes, 3, clamp,
            moving_average_mode, moving_average_momentum, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.nonlin2 = non_linearity_map[nonlins[1]]()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        """Forward pass of RegularBasicBlock."""
        out = self.nonlin1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        out = self.nonlin2(out)
        return out


class XnorBasicBlock(nn.Module):
    """
    ResNet XNOR regular basic block.

    Block structure (BN -> Quant -> Conv -> NonLin):

        Rastegari, Mohammad, et al.
        "Xnor-net: Imagenet classification using binary convolutional neural networks."
        European conference on computer vision. Springer, Cham, 2016.

    Using double shortcuts:

        Zechun Liu, Baoyuan Wu, Wenhan Luo, Xin Yang, Wei Liu, and Kwang-Ting Cheng.
        "Bi-real net: Enhancing the performance of 1-bit CNNs with improved representational
        capability and advanced training algorithm."
        In Proceedings of the European conference on computer vision (ECCV), pages 722–737, 2018.
    """

    def __init__(
        self, in_planes: int, planes: int, x_quant: str, w_quant: str,
        nonlins: List[str], stride: int = 1, double_shortcut: bool = False,
        clamp: Optional[Dict] = None,
        moving_average_mode: str = 'off',
        moving_average_momentum: float = 0.99,
    ) -> None:
        """
        Build ResNet XNOR basic block.

        Args:
            in_planes: the number of in-channels for the block
            planes: the number of out-channels for the block
            x_quant: quantization scheme for activations,
                see :mod:`~quant.binary.binary_conv`.
            w_quant: quantization scheme for weights,
                see :mod:`~quant.binary.binary_conv`.
            nonlins: non-linearities for the block. It should be a list of two
                strings, where each string is in {'relu', 'prelu', 'identity'}.
            stride: stride size
            double_shortcut: whether to use double shortcuts.
            clamp: clamping scheme for activations.
                It should have a key named "kind" indicating the kind of clamping function
                and other keys indicating other potential arguments.
                See :mod:`~quant.binary.binary_conv`.
            moving_average_mode: moving average mode to use,
                see :class:`~quant.binary.activation_quantization.ActivationQuantizer`.
            moving_average_momentum: momentum for moving average update,
                see :class:`~quant.binary.activation_quantization.ActivationQuantizer`.
        """
        super(XnorBasicBlock, self).__init__()
        if len(nonlins) != 2:
            raise ValueError('There should be 2 non-linearities.')
        self.double_shortcut = double_shortcut

        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = QuantConv2d(
            x_quant, w_quant, in_planes, planes, 3, clamp,
            moving_average_mode, moving_average_momentum, stride=stride, padding=1, bias=True
        )
        self.nonlin1 = non_linearity_map[nonlins[0]]()

        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = QuantConv2d(
            x_quant, w_quant, planes, planes, 3, clamp,
            moving_average_mode, moving_average_momentum, stride=1, padding=1, bias=True
        )
        self.nonlin2 = non_linearity_map[nonlins[1]]()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    planes,
                    kernel_size=1,
                    stride=stride,
                    bias=True,
                ),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        """Forward pass of XnorBasicBlock."""
        out1 = self.nonlin1(self.conv1(self.bn1(x)))
        if self.double_shortcut:
            out1 = out1 + self.shortcut(x)
        out2 = self.conv2(self.bn2(out1))
        if self.double_shortcut:
            out2 = self.nonlin2(out2)
            return out2 + out1
        out2 = out2 + self.shortcut(x)
        return self.nonlin2(out2)


class QResNet(nn.Module):
    """
    ResNet implementation supporting full precision and quantized schemes.

    Note we use full-precision down-sampling. See:

        Zechun Liu, Baoyuan Wu, Wenhan Luo, Xin Yang, Wei Liu, and Kwang-Ting Cheng.
        Bi-real net: Enhancing the performance of 1-bit CNNs with improved representational
        capability and advanced training algorithm.
        In Proceedings of the European conference on computer vision (ECCV), pages 722–737, 2018.

    Two types of blocks can be used, either
    :class:`~quant.models.resnet.RegularBasicBlock` (regular) or
    :class:`~quant.models.resnet.XnorBasicBlock` (xnor).

    ResNet consists of the following layers:
    layer0 (first layer), layer1, layer2, layer3, layer4 (optional), layer5 (last layer).

    `layer0` is the feature extractor layer (conv1).
    Its config dictionary contains keys: `n_in_channels`, `kernel_size`, `stride`, `padding`,
    `bias`, and `maxpool`.
    It is important to note that `n_in_channels` does not refer to the number of channels of the
    image (3), but rather the number of input channels to `layer1`.
    All arguments except for `maxpool` are passed to PyTorch ``nn.Conv2d``.
    `maxpool` is another dictionary with keys `type`, `kernel_size`, `stride`, and `padding`.
    If the type is `identity`, there is no pooling.
    If the type is `maxpool2d`, then the other keys are passed to construct ``nn.MaxPool2d``.

    `layer1`, `layer2`, `layer3`, `layer4` are all dictionaries used to
    configure the corresponding layers.
    Usually they can all be the same dictionary.
    The keys and values here are used to construct either
    :class:`~quant.models.resnet.RegularBasicBlock` or :class:`~quant.models.resnet.XnorBasicBlock`
    depending on what is specified in `block`.

    `nonlins` is a list of two strings specifying the non-linearity to use inside each layer.
    Each string value can be `relu`, `prelu`, or `identity`.
    """

    def __init__(
        self,
        loss_fn: Callable[..., torch.Tensor],
        block: str,
        layer0: dict,
        layer1: dict,
        layer2: dict,
        layer3: dict,
        layer4: Optional[dict],
        nonlins: List[str],
        num_blocks: List[int],
        output_classes: int,
        moving_average_mode: str = 'off',
        moving_average_momentum: float = 0.99,
    ) -> None:
        """
        Construct QResNet.

        Args:
            loss_fn: loss function of the model
            block: name of the block to use ('regular' or 'xnor')
            layer0: configuration for conv1 layer of the model
            layer1: configuration for layer1 layer of the model
            layer2: configuration for layer2 layer of the model
            layer3: configuration for layer3 layer of the model
            layer4: configuration for layer4 layer of the model
            nonlins: non-linearities to use for each layer. It should be a list of two
                strings, where each string is in {'relu', 'prelu', 'identity'}.
            num_blocks: a list representing the number of blocks in each layer
            output_classes: number of output classes
            moving_average_mode: moving average mode to use
                see :class:`~quant.binary.activation_quantization.ActivationQuantizer`.
            moving_average_momentum: momentum for moving average
                update, see :class:`~quant.binary.activation_quantization.ActivationQuantizer`.
        """
        super(QResNet, self).__init__()
        # loss_fn is a loss function in torch.nn.functional
        setattr(self, 'loss_fn', loss_fn)

        blocks = {
            'regular': RegularBasicBlock,
            'xnor': XnorBasicBlock,
        }
        try:
            block_cls: Union[RegularBasicBlock, XnorBasicBlock] \
                = blocks[block]  # type: ignore
        except KeyError:
            raise ValueError(f'Block {block} is not supported.')

        n_in_channels = layer0['n_in_channels']

        self.conv1 = nn.Conv2d(
            3,
            n_in_channels,
            kernel_size=layer0['kernel_size'],
            stride=layer0['stride'],
            padding=layer0['padding'],
            bias=layer0['bias'],
        )
        if layer0['maxpool']['type'] == 'identity':
            self.maxpool = nn.Identity()
        elif layer0['maxpool']['type'] == 'maxpool2d':  # pragma: no cover (coverage does not report it even though it's covered)  # noqa: E501
            self.maxpool = nn.MaxPool2d(  # type: ignore
                kernel_size=layer0['maxpool']['kernel_size'],
                stride=layer0['maxpool']['stride'],
                padding=layer0['maxpool']['padding'],
            )
        else:
            raise ValueError(
                f"maxpool type {layer0['maxpool']['type']} is not supported."
            )

        self.bn1 = nn.BatchNorm2d(n_in_channels)

        self.blocks = nn.ModuleList(
            [nn.Sequential(self.conv1, self.bn1, nn.ReLU(inplace=True), self.maxpool)]
        )

        n_planes = self._make_layer(
            block_cls, layer1,
            n_in_channels, n_in_channels, num_blocks[0], nonlins, stride=1,
            moving_average_mode=moving_average_mode,
            moving_average_momentum=moving_average_momentum
        )
        n_planes = self._make_layer(
            block_cls, layer2,
            n_planes, 2 * n_in_channels, num_blocks[1], nonlins, stride=2,
            moving_average_mode=moving_average_mode,
            moving_average_momentum=moving_average_momentum
        )
        n_planes = self._make_layer(
            block_cls, layer3,
            n_planes, 4 * n_in_channels, num_blocks[2], nonlins, stride=2,
            moving_average_mode=moving_average_mode,
            moving_average_momentum=moving_average_momentum
        )
        if layer4 is not None:  # pragma: no cover (coverage does not report it even though it's covered)  # noqa: E501
            n_planes = self._make_layer(
                block_cls, layer4,
                n_planes, 8 * n_in_channels, num_blocks[3], nonlins, stride=2,
                moving_average_mode=moving_average_mode,
                moving_average_momentum=moving_average_momentum
            )

        self.linear_classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),  # type: ignore
            nn.Linear(n_planes, output_classes)
        )

    def _make_layer(
        self,
        block: Union[RegularBasicBlock, XnorBasicBlock],
        layer_config: dict,
        in_planes: int,
        out_planes: int,
        num_blocks: int,
        nonlins: List[str],
        stride: int,
        moving_average_mode: str = 'off',
        moving_average_momentum: float = 0.99,
    ) -> int:
        """
        Make a layer (layer1, layer2, layer3, layer4).

        Args:
            block:
                block to user in the layer
            layer_config: a dictionary containing the config for the layer.
                It should have the following keys:
                * x_quant: quantization scheme for activations
                * w_quant: quantization scheme for weights
                * clamp: clamping scheme for activations.
                It should have a key named "kind" indicating the kind of clamping function
                and other keys indicating other potential arguments.
                * other optional keys such as double_shortcut
            in_planes: the number of in-channels for the layer
            out_planes: the number of out-channels for the layer
            num_blocks: the number of blocks for the layer
            nonlins: non-linearities for the current layer. It should be a list of two
                strings, where each string is in {'relu', 'prelu', 'identity'}.
            stride: stride size
            moving_average_mode: moving average mode to use
                see :class:`~quant.binary.activation_quantization.ActivationQuantizer`.
            moving_average_momentum: momentum for moving average
                update, see :class:`~quant.binary.activation_quantization.ActivationQuantizer`.

        Returns:
            the number of planes of the layer
        """
        strides = [stride] + [1] * (num_blocks - 1)
        for stride in strides:
            self.blocks.append(
                block(in_planes, out_planes, nonlins=nonlins, stride=stride,
                      moving_average_mode=moving_average_mode,
                      moving_average_momentum=moving_average_momentum, **layer_config)
            )
            in_planes = out_planes

        return in_planes

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        """Forward pass of XnorBasicBlock."""
        for block in self.blocks:
            x = block(x)
        return self.linear_classifier(x)
