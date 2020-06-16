#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

"""
Utilities for supporting visualization with TensorBoard.

Quant supports visualizing loss and evaluation metrics during training in TensorBoard.
"""

from typing import List, Tuple
from functools import partial
from pathlib import Path
import shutil
from typing import Any, Dict, Optional

from torch.utils.tensorboard import SummaryWriter

from quant.common.metrics import Metric


class Visualizer:
    """TensorBoard visualizer."""

    def __init__(
        self,
        tensorboard_base_dir: Path,
        root_experiments_dir: Path,
        experiment_name: str,
    ) -> None:
        """
        Create a visualizer object for TensorBoard.

        Args:
            tensorboard_base_dir: Root directory where TensorBoard experiments are stored
            root_experiments_dir: Root directory for storing logs, checkpoints, etc.
            experiment_name: Name of the experiment
        """
        self.tensorboard_base_dir = tensorboard_base_dir
        self.root_experiments_dir = root_experiments_dir
        self.experiment_name = experiment_name
        self.writer = SummaryWriter(str(tensorboard_base_dir / experiment_name))  # type: ignore

    def hook(self, split: str, metrics: Dict[str, Metric],
             epoch: int, global_step: int, log_interval: int = 10,
             values_dict: Optional[Dict[str, float]] = None, **kwargs: Any) -> None:
        """
        Provide a training / test loop-compatible hook for logging evaluation metrics.

        Args:
            split: The split to visualize, e.g. train or test
            metrics: Dictionary mapping metric names to Metric objects
            epoch: Training epoch
            global_step: Unique incrementing integer across all epochs indicating the step
            log_interval: frequency for logging metrics
            values_dict: Dictionary mapping names to values
                for other non-metric values to log
        """
        if values_dict is None:
            values_dict = {}

        if split != 'train':
            for name, metric in metrics.items():
                name = name.replace(' ', '_')
                self.writer.add_scalar(f'{name}/{split}', metric.compute(), epoch)

            for name, val in values_dict.items():
                name = name.replace(' ', '_')
                self.writer.add_scalar(f'{name}/{split}', val, epoch)

        elif global_step % log_interval == 0:
            for name, metric in metrics.items():
                name = name.replace(' ', '_')
                self.writer.add_scalar(f'{name}/{split}', metric.compute(), global_step)

            for name, val in values_dict.items():
                name = name.replace(' ', '_')
                self.writer.add_scalar(f'{name}/{split}', val, global_step)

    def __del__(self) -> None:
        """Make a copy of the summary writer logs in the experiment artifacts."""
        shutil.copytree(
            self.tensorboard_base_dir / self.experiment_name,
            self.root_experiments_dir / self.experiment_name / 'tensorboard',
        )


def get_tensorboard_hooks(
    config: dict, experiment_root_directory: Path,
    train_metrics: Dict[str, Metric], test_metrics: Dict[str, Metric]
) -> Tuple[List, List]:
    """
    Get TensorBoard hooks for visualizing metrics as training progresses.

    Args:
        config: experiment config
        experiment_root_directory: root directory for storing logs, checkpoints, etc.
        train_metrics: dict mapping metric keys to metric objects for training
        test_metrics: dict mapping metric keys to metric objects for testing
    """
    log_config = config['log']

    train_hooks = []
    test_hooks = []

    if log_config['tensorboard']:
        visualizer = Visualizer(
            Path(log_config['tensorboard_root']),
            Path(experiment_root_directory),
            config['experiment_name'],
        )
        train_hooks.append(partial(visualizer.hook, split='train', metrics=train_metrics))
        test_hooks.append(partial(visualizer.hook, split='test', metrics=test_metrics))

    return train_hooks, test_hooks
