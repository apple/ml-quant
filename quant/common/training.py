#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

"""
Quant provides generic training and test loops that can be used for all datasets.

Training and test loops both support hooks, which are functions that are called
inside each batch of each epoch. This allows different driver scripts to use
the same training and test loops, sharing the same structure, while making it possible
to introduce custom behavior.

Each hook can take a variable number of keyword arguments,
They will always be given `epoch` and `global_step`.
`epoch` is an integer, starting from 1, that represents the current epoch.
`global_step` is a unique, incrementing counter for every batch of every epoch.
It starts at `1` and goes to `num_epochs * ceil(dataset_size / batch_size)`, inclusive.
Hooks can use the other keyword arguments to implement custom behavior.

One example of a hook implemented in the library is the visualization hook
that supports logging metrics to be viewed via TensorBoard:
:meth:`quant.common.visualization.Visualizer.hook`
"""

import logging
from typing import Dict, Optional, Sequence, Union

import torch
import torch.nn as nn
from torch.optim import Optimizer  # type: ignore
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data.dataloader import DataLoader

from quant import Hook
from quant.common.metrics import Metric


logger = logging.getLogger(__name__)


def _get_lr(optimizer: Optimizer) -> float:
    """
    Get learning rate of the first parameter group.

    Args:
        optimizer (optim.Optimizer): PyTorch optimizer
    """
    for param_group in optimizer.param_groups:
        return param_group['lr']

    raise ValueError('Cannot get optimizer LR: optimizer does not have any parameter groups.')


def project(optimizer: Optimizer) -> None:
    """Project model parameters to a range so that they can be updated."""
    # No-op
    # In theory, we should project the quantized weights to the [-1, 1] range
    # so that we have non-zero gradients and they can be updated.
    # However, in practice, we notice that this does not make a difference.
    # Hence, this is a no-op.
    _ = optimizer
    return None


def train(
    model: Union[nn.Module, nn.DataParallel],
    train_loader: DataLoader,
    metrics: Dict[str, Metric],
    optimizer: Optimizer,
    scheduler: _LRScheduler,
    device: torch.device,
    epoch: int,
    log_interval: int,
    hooks: Optional[Sequence[Hook]] = None,
    teacher: Optional[Union[nn.Module, nn.DataParallel]] = None,
) -> Dict[str, float]:
    """
    Train a model on some data using some criterion and with some optimizer.

    Args:
        model: Model to train
        train_loader: Data loader for loading training data
        metrics: A dict mapping evaluation metric names to metrics classes
        optimizer: PyTorch optimizer
        scheduler: PyTorch scheduler
        device: PyTorch device object
        epoch: Current epoch, where the first epoch should start at 1
        log_interval: Number of batches before printing loss
        hooks: A sequence of functions that can implement custom behavior
        teacher: teacher network for knowledge distillation, if any

    Returns:
        A dictionary mapping evaluation metric names to computed values for the training set.
    """
    if hooks is None:
        hooks = []

    model.train()
    for metric in metrics.values():
        metric.reset()

    loss_fn = model.module.loss_fn if isinstance(model, nn.DataParallel) else model.loss_fn

    seen_examples = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        if teacher is None:
            teacher_output = None
            loss = loss_fn(output, target)  # type: ignore
        else:
            teacher_output = teacher(data)
            loss = loss_fn(output, teacher_output, target)  # type: ignore
        loss.backward()
        optimizer.step()
        project(optimizer)
        scheduler.step()  # type: ignore

        with torch.no_grad():
            for metric in metrics.values():
                metric.update(output, target, teacher_output=teacher_output)

        for hook in hooks:
            hook(
                epoch=epoch,
                global_step=1 + (epoch - 1) * len(train_loader.dataset) + batch_idx,
                values_dict={'lr': _get_lr(optimizer)},
                log_interval=log_interval,
            )

        seen_examples += len(data)
        if batch_idx % log_interval == 0:
            logger.info(
                'Train Epoch: {} [{}/{} ({:.0f}%)]\tBatch Loss: {:.6f}'.format(
                    epoch,
                    seen_examples,
                    len(train_loader.dataset),
                    100 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )

    # Computing evaluation metrics for training set
    computed_metrics = {name: metric.compute() for name, metric in metrics.items()}

    logger.info('Training set evaluation metrics:')
    for name, metric in metrics.items():
        logger.info(f'{name}: {metric}')

    return computed_metrics


def evaluate(
    model: Union[nn.Module, nn.DataParallel],
    test_loader: DataLoader,
    metrics: Dict[str, Metric],
    device: torch.device,
    epoch: int,
    hooks: Optional[Sequence[Hook]] = None,
) -> Dict[str, float]:
    """
    Evaluate model on some held-out set.

    Args:
        model: Model to test on
        test_loader: Data loader for loading test data
        metrics: A dict mapping evaluation metric names to metrics classes
        device: PyTorch device object
        epoch: Current epoch, where the first epoch should start at 1
        hooks: A sequence of functions that can implement custom behavior

    Returns:
        A dictionary mapping evaluation metric names to computed values.
    """
    if hooks is None:
        hooks = []

    model.eval()
    for metric in metrics.values():
        metric.reset()

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)

            for metric in metrics.values():
                metric.update(output, target)

    for hook in hooks:
        hook(
            epoch=epoch,
            global_step=1 + (epoch - 1) * len(test_loader.dataset) + batch_idx
        )

    computed_metrics = {name: metric.compute() for name, metric in metrics.items()}

    logger.info('Test set evaluation metrics:')
    for name, metric in metrics.items():
        logger.info(f'{name}: {metric}')

    return computed_metrics
