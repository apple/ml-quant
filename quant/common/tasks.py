#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

"""Utilities for running tasks."""

from functools import partial
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Type, Union
import yaml

import torch
import torch.nn as nn

from quant import Hook, MetricDict
from quant.common import init_logging
from quant.utils.checkpoints import get_path_to_checkpoint, log_checkpoints, \
    restore_from_checkpoint
from quant.common.initialization import (
    get_device,
    get_model,
    get_optimizer,
    get_lr_scheduler,
    get_loss_fn,
)
from quant.common.metrics import LossMetric, Top1Accuracy, TopKAccuracy
from quant.common.training import train, evaluate
from quant.data.data_loaders import QuantDataLoader
from quant.utils.kd_criterion import kd_criterion


def get_teacher_and_kd_loss(
    teacher_config_path: str,
    teacher_checkpoint_path: str,
    train_mode: bool,
    criterion_config: dict,
    device: torch.device,
    ngpus: int,
    freeze_teacher: bool = True,
    strict_keys: bool = True,
) -> Tuple[Union[nn.Module, nn.DataParallel], Callable[..., torch.Tensor]]:
    """
    Get teacher and KD loss for knowledge distillation.

    Args:
        teacher_config_path: path to config used to train teacher
        teacher_checkpoint_path: path to checkpoint to use to initialize teacher
        train_mode: if true, use teacher in train mode, or use eval mode otherwise
        criterion_config: config for KD criterion, such as alpha and temperature
        device: PyTorch device used to store teacher, should the be the same as model
        ngpus: number of GPUs to run teacher, should be the same as that of the student model
        freeze_teacher: whether to freeze teacher
        strict_keys: whether to enforce keys must exactly match for restoring checkpoint

    Returns:
        An initialized teacher and KD loss function with teacher-related args resolved
    """
    with open(teacher_config_path) as f:
        teacher_config = yaml.safe_load(f)
        teacher_model_config = teacher_config['model']

    loss_fn = get_loss_fn(teacher_model_config['loss'])
    teacher = get_model(
        architecture=teacher_model_config['architecture'],
        loss_fn=loss_fn,
        arch_config=teacher_model_config['arch_config'],
        device=device,
        ngpus=ngpus,
    )

    restore_from_checkpoint(teacher, None, None, teacher_checkpoint_path, device, strict_keys)

    if freeze_teacher:
        for p in teacher.parameters():
            p.requires_grad_(False)

    teacher.train() if train_mode else teacher.eval()

    kd_loss = partial(kd_criterion, freeze_teacher=freeze_teacher, **criterion_config)

    return teacher, kd_loss


def classification_task(
    config: dict,
    experiment_root_directory: Path,
    data_loader_cls: Type[QuantDataLoader],
    get_hooks: Callable[[dict, Path, MetricDict, MetricDict], Tuple[List[Hook], List[Hook]]],
    restore_experiment: Optional[Path] = None,
) -> Tuple[List[Dict[str, float]], List[Dict[str, float]]]:
    """
    Driver program for running classification task.

    Args:
        config: merged config with CLI args
        experiment_root_directory: root directory for storing logs, checkpoints, etc.
        data_loader_cls: The QuantDataLoader class
        get_hooks: a function that returns lists of training and testing hooks
        restore_experiment: path to experiment to restore, None for do not restore

    Returns:
        (List of training set metrics for each epoch, list of test set metrics for each epoch).
    """
    env_config = config['environment']
    data_config = config['data']
    model_config = config['model']
    optimization_config = config['optimization']
    log_config = config['log']

    init_logging(log_config['level'])

    device = get_device(env_config['ngpus'], config.get('seed'), **env_config.get('cuda', {}))

    data_loader = data_loader_cls(**data_config)
    train_loader = data_loader.get_train_loader() if not config.get('skip_training') else None
    test_loader = data_loader.get_test_loader()

    epochs = optimization_config['epochs']

    teacher = None
    use_kd = 'kd_config' in model_config
    if use_kd:
        teacher, kd_loss = get_teacher_and_kd_loss(
            device=device, ngpus=env_config['ngpus'],
            strict_keys=model_config.get('strict_keys', True),
            **model_config['kd_config']
        )

    loss_fn = get_loss_fn(model_config['loss']) if not use_kd else kd_loss
    model = get_model(
        architecture=model_config['architecture'],
        loss_fn=loss_fn,
        arch_config=model_config['arch_config'],
        device=device,
        ngpus=env_config['ngpus'],
    )

    optimizer, scheduler = None, None
    if not config.get('skip_training'):
        optimizer = get_optimizer(model.parameters(), optimization_config['optimizer'])
        scheduler = get_lr_scheduler(optimizer, optimization_config['lr_scheduler'], epochs, len(train_loader))  # type: ignore  # noqa: E501

    if restore_experiment is not None:
        checkpoint_path = get_path_to_checkpoint(restore_experiment)
        model, restored_optimizer, restored_scheduler, start_epoch = restore_from_checkpoint(
            model,
            optimizer,
            scheduler,
            checkpoint_path,
            device,
            model_config.get('strict_keys', True),
        )
        optimizer, scheduler = restored_optimizer, restored_scheduler
        start_epoch += 1
    elif config.get('init_from_checkpoint'):
        model, _, _, _ = restore_from_checkpoint(
            model,
            None,
            None,
            config['init_from_checkpoint'],
            device,
            model_config.get('strict_keys', True),
        )
        start_epoch = 1
    else:
        start_epoch = 1

    train_metrics = {
        'Loss': LossMetric(loss_fn, accumulate=True),
        'Top-1 Accuracy': Top1Accuracy(accumulate=True),
        'Top-5 Accuracy': TopKAccuracy(5, accumulate=True),
    }

    test_metrics = {
        'Loss': LossMetric(get_loss_fn(model_config['loss']), accumulate=True),
        'Top-1 Accuracy': Top1Accuracy(accumulate=True),
        'Top-5 Accuracy': TopKAccuracy(5, accumulate=True),
    }

    train_hooks, test_hooks = get_hooks(config, experiment_root_directory,
                                        train_metrics, test_metrics)
    train_epoch_metrics, test_epoch_metrics = [], []

    if config.get('skip_training'):
        computed_test_metrics = evaluate(
            model=model,
            test_loader=test_loader,
            metrics=test_metrics,
            device=device,
            epoch=1,
            hooks=test_hooks,
        )
        test_epoch_metrics.append(computed_test_metrics)
    else:
        for epoch in range(start_epoch, start_epoch + epochs):
            computed_train_metrics = train(
                model=model,
                train_loader=train_loader,  # type: ignore
                metrics=train_metrics,
                optimizer=optimizer,
                scheduler=scheduler,  # type: ignore
                device=device,
                epoch=epoch,
                log_interval=log_config['interval'],
                hooks=train_hooks,
                teacher=teacher,
            )
            computed_test_metrics = evaluate(
                model=model,
                test_loader=test_loader,
                metrics=test_metrics,
                device=device,
                epoch=epoch,
                hooks=test_hooks,
            )

            train_epoch_metrics.append(computed_train_metrics)
            test_epoch_metrics.append(computed_test_metrics)

            if epoch % log_config['save_model_freq'] == 0 or epoch == epochs:
                log_checkpoints(
                    experiment_root_directory / config['experiment_name'] / 'checkpoints',
                    model,
                    optimizer,  # type: ignore
                    scheduler,  # type: ignore
                    epoch,
                )

    data_loader.cleanup()

    return train_epoch_metrics, test_epoch_metrics
