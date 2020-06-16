#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

"""Utilities for initializing device, model, optimizer, and LR scheduler."""

import copy
from typing import Callable, Dict, Iterator, List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from quant.utils.linear_lr_scheduler import LinearLR
from quant.models.lenet import QLeNet5
from quant.models.resnet import QResNet

model_mapping = {
    'lenet5': QLeNet5,
    'resnet': QResNet,
}


def get_loss_fn(loss: str) -> Callable[..., torch.Tensor]:
    """
    Get loss function as a PyTorch functional loss based on the name of the loss function.

    Choices include 'cross_entropy', 'nll_loss', and 'kl_div'.

    Args:
        loss: a string indicating the loss function to return.
    """
    loss_fn_mapping: Dict[str, Callable[..., torch.Tensor]] = {
        'cross_entropy': F.cross_entropy,
        'nll_loss': F.nll_loss,
        'kl_div': F.kl_div,
    }

    try:
        loss_fn: Callable[..., torch.Tensor] = loss_fn_mapping[loss]
    except KeyError:
        raise ValueError(f'Loss function {loss} is not supported.')

    return loss_fn


def get_device(
    ngpus: int,
    seed: int = None,
    cudnn_deterministic: bool = False,
    cudnn_benchmark: bool = False,
) -> torch.device:
    """
    Initialize PyTorch device and sets random seed.

    Args:
        ngpus: Number of GPUs, 0 for CPU
        seed: initial random seed for reproducibility
        cudnn_deterministic: make CUDNN deterministic
        cudnn_benchmark: use CUDNN auto-tuner

    Returns:
        A PyTorch device object.
    """
    use_cuda = ngpus > 0 and torch.cuda.is_available()

    if seed:
        torch.manual_seed(seed)  # type: ignore

    if use_cuda:  # pragma: no cover
        torch.backends.cudnn.deterministic = cudnn_deterministic  # type: ignore
        torch.backends.cudnn.benchmark = cudnn_benchmark  # type: ignore
        best_gpu_device_id = _get_best_gpus(1)[0]
        # For data parallelism, parameters and buffers must be stored on the 1st device, devices[0]
        # Here we ensure that we always return the first device id from the
        # device ids available for DataParallel
        device = torch.device(f'cuda:{best_gpu_device_id}')
    else:
        device = torch.device('cpu')

    return device


def _get_best_gpus(k: int) -> List[int]:
    """Return the top k device ids associated with GPUs with the best compute capability."""
    # Select top ngpus based on CUDA device capability score
    max_gpus = torch.cuda.device_count()
    capabilities = [torch.cuda.get_device_capability(i) for i in range(max_gpus)]
    ranked_device_ids = sorted(enumerate(capabilities), key=lambda t: t[1], reverse=True)
    device_ids = [d[0] for d in ranked_device_ids][:k]
    return device_ids


def get_model(
    architecture: str, loss_fn: Callable[..., torch.Tensor],
    arch_config: dict, device: torch.device, ngpus: int
) -> Union[nn.Module, nn.DataParallel]:
    """
    Get model from config.

    Args:
        architecture: model architecture
        loss_fn: loss function in ``torch.nn.functional``
        arch_config: architecture config to be passed to model constructor
        device: the device this model should be stored on
        ngpus: the number of GPUs to use

    Returns:
        A nn.Module object if for single GPU, or nn.DataParallel object if using multiple GPUs
    """
    try:
        model = model_mapping[architecture](loss_fn=loss_fn, **arch_config)
    except KeyError:
        raise ValueError(f'Model architecture {architecture} is not found.')

    max_gpus = torch.cuda.device_count()
    if ngpus > max_gpus:
        raise ValueError(
            f"Device only has {max_gpus} GPUs, but {ngpus} are specified."
        )

    if ngpus > 1:
        best_gpus = _get_best_gpus(ngpus)
        model = nn.DataParallel(model, device_ids=best_gpus)

    model = model.to(device)

    return model


def get_optimizer(parameters: Iterator[nn.Parameter], config: dict) -> optim.Optimizer:  # type: ignore  # noqa: E501
    """
    Get an optimizer.

    Choices include 'sgd', 'adam', and 'sgd'.

    Args:
        parameters: Parameters to optimize
        config: A dictionary containing configurations for the optimizer.
            It must have at minimum an 'algorithm' key and
            `required arguments <https://pytorch.org/docs/stable/optim.html#algorithms/>`_
            for the optimizer.

    Returns:
        A PyTorch optimizer.
    """
    config = copy.deepcopy(config)
    algorithm = config.pop('algorithm')

    name_to_optimizer = {
        'adadelta': optim.Adadelta,  # type: ignore
        'adam': optim.Adam,
        'sgd': optim.SGD,
    }

    return name_to_optimizer[algorithm](parameters, **config)


def get_lr_scheduler(
    optimizer: optim.Optimizer, config: dict, epochs: int, steps_per_epoch: int  # type: ignore
) -> optim.lr_scheduler._LRScheduler:
    """
    Get a LR scheduler.

    Choices include 'step_lr', 'multi_step_lr', 'linear_lr', and 'lambda_lr'.

    Typically in PyTorch, the learning rate scheduler calls `step()` after every epoch.
    In this project, we call `step()` after every batch in every epoch.
    Hence, parameters such as `step_lr` in `StepLR` and `milestones` in `MultiStepLR`
    are scaled by the number of steps per epoch.
    If you use `LambdaLR`, keep in mind that the lambda function takes the
    global step (batch) index, not the epoch index.

    We have one custom learning rate scheduler,
    :class:`~quant.common.linear_lr_scheduler.LinearLR`, that can be used by selecting `linear_lr`.

    All other schedulers are shipped with PyTorch.

    Args:
        optimizer: Optimizer to adjust learning rate for
        config: A dictionary containing configurations for the LR scheduler.
            It must have at minimum a 'scheduler' key and
            `args <https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate>`_
            for the scheduler.
        epochs: total number of epochs
        steps_per_epoch: Steps (batches) per epoch

    Returns:
        A PyTorch learning rate scheduler.
    """
    config = copy.deepcopy(config)
    scheduler = config.pop('scheduler')

    name_to_scheduler = {
        'linear_lr': LinearLR,
        'lambda_lr': lr_scheduler.LambdaLR,
        'step_lr': lr_scheduler.StepLR,
        'multi_step_lr': lr_scheduler.MultiStepLR,
    }

    if scheduler == 'linear_lr':
        config['steps_per_epoch'] = steps_per_epoch
        config['total_epochs'] = epochs
        config['min_lr'] = float(config['min_lr'])  # YAML parses 2e-7 to a string instead of float
    elif scheduler == 'lambda_lr':
        config['lr_lambda'] = eval(config['lr_lambda'])
    elif scheduler == 'step_lr':
        config['step_size'] *= steps_per_epoch
    elif scheduler == 'multi_step_lr':  # pragma: no cover (coverage does not report it even though it's covered)  # noqa: E501
        new_milestones = [epochs * steps_per_epoch for epochs in config['milestones']]
        config['milestones'] = new_milestones

    return name_to_scheduler[scheduler](optimizer, **config)
