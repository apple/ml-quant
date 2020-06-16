#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

"""Utilities for working with checkpoints."""

from pathlib import Path
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.optimizer import Optimizer


def log_checkpoints(
    checkpoint_dir: Path,
    model: Union[nn.Module, nn.DataParallel],
    optimizer: Optimizer,
    scheduler: optim.lr_scheduler._LRScheduler,
    epoch: int,
) -> None:
    """
    Serialize a PyTorch model in the `checkpoint_dir`.

    Args:
        checkpoint_dir: the directory to store checkpoints
        model: the model to serialize
        optimizer: the optimizer to be saved
        scheduler: the LR scheduler to be saved
        epoch: the epoch number
    """
    checkpoint_file = 'checkpoint_{}.pt'.format(epoch)
    checkpoint_dir.mkdir(exist_ok=True, parents=True)
    file_path = checkpoint_dir / checkpoint_file

    if isinstance(model, nn.DataParallel):
        model_state_dict = model.module.state_dict()
    else:
        model_state_dict = model.state_dict()

    torch.save(  # type: ignore
        {
            'epoch': epoch,
            'model_state_dict': model_state_dict,
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
        },
        file_path,
    )


def restore_from_checkpoint(
    model: Union[nn.Module, nn.DataParallel],
    optimizer: Optional[Optimizer],
    scheduler: Optional[optim.lr_scheduler._LRScheduler],
    checkpoint_path: str,
    device: torch.device,
    strict_keys: bool = True,
) -> Tuple[
    Union[nn.Module, nn.DataParallel],
    Optional[Optimizer],
    Optional[optim.lr_scheduler._LRScheduler],
    int,
]:
    """
    Restore model, optimizer, and learning rate scheduler state from checkpoint.

    Args:
        model: the model object to be restored
        optimizer: the optimizer to be restored
        scheduler: the LR scheduler to be restored
        checkpoint_path: path to a model checkpoint
        device: the device to load data to. Note that
            the model could be saved from a different device.
            Here we transfer the parameters to the current given device.
            So, a model could be trained and saved on GPU, and be loaded on CPU, for example.
        strict_keys: If True keys in state_dict should be identical after restoring

    Returns:
        the initialized model, optimizer, scheduler, and epoch from the checkpoint
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)   # type: ignore
    if isinstance(model, nn.DataParallel):
        model.module.load_state_dict(checkpoint['model_state_dict'], strict=strict_keys)
    else:
        model.load_state_dict(checkpoint['model_state_dict'], strict=strict_keys)

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Transfer parameters internal variable to models device
        for state in optimizer.state.values():  # type: ignore
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

    if scheduler is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    loaded_epoch = checkpoint['epoch']

    return model, optimizer, scheduler, loaded_epoch


def get_path_to_checkpoint(experiment_path: Path, epoch: Optional[int] = None) -> str:
    """
    Find checkpoint file path in an experiment directory.

    Assume that checkpoint file names follow the `checkpoint_{epoch}.pt` format.

    Args:
        experiment_path: path to an experiment directory
        epoch: If given tries to load that checkpoint, otherwise
            loads the last checkpoint

    Returns:
        Path to checkpoint file
    """
    ckpts_path = experiment_path / 'checkpoints'
    ckpts_dict = {
        int(path.name.split('_')[1].split('.')[0]): path
        for path in ckpts_path.iterdir()
    }
    if len(ckpts_dict) == 0:
        raise ValueError(
            f'No checkpoint exists in the experiment directory: {experiment_path}'
        )
    if epoch is not None:
        if epoch not in ckpts_dict.keys():
            raise ValueError(f'Could not find checkpoint for epoch {epoch}.')
    else:
        epoch = max(ckpts_dict.keys())

    return str(ckpts_dict[epoch])
