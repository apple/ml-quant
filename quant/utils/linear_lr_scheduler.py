#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

"""Linear PyTorch learning rate scheduler."""

from typing import List

from torch.optim import Optimizer  # type: ignore
from torch.optim.lr_scheduler import _LRScheduler


class LinearLR(_LRScheduler):
    """Decays the learning rate following a linear schedule."""

    def __init__(self, optimizer: Optimizer, min_lr: float, total_epochs: int,
                 steps_per_epoch: int, last_epoch: int = -1) -> None:
        """
        Construct a linear lr scheduler specifying the minimum lr and last epoch.

        Args:
            optimizer:  Wrapped optimizer.
            min_lr: Minimum learning rate.
            total_epochs: Total number of epochs.
            steps_per_epoch: The number of steps (batches) per epoch.
            last_epoch: The index of the last batch.
                This parameter is used when resuming a training job.
                Since step() should be invoked after each batch instead of after each epoch,
                this number represents the total number of batches computed,
                not the total number of epochs computed.
                When last_epoch=-1, the schedule is started from the beginning. Default: -1
                The batch size should be consistent between the resuming job and the prior job,
                or else the scheduler can be wrong.

        """
        self.min_lr = min_lr
        self.total_epochs = total_epochs
        self.steps_per_epoch = steps_per_epoch
        super(LinearLR, self).__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:  # type: ignore
        """Get the current learning rate for each parameter group."""
        lrs = []
        total_steps = (self.total_epochs - 1) * self.steps_per_epoch
        last_epoch = self.last_epoch  # type: ignore
        for group in self.optimizer.param_groups:  # type: ignore
            lr_0 = group['initial_lr']
            lr = max(
                lr_0 - last_epoch / total_steps * (lr_0 + self.min_lr), self.min_lr
            )
            lrs.append(lr)

        return lrs
