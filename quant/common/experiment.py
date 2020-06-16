#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

"""
An experiment represents a single run of a task (model + data) in some configuration.

An experiment is always run on some :class:`quant.common.compute_platform.ComputePlatform`.
It produces artifacts that can be used to reproduce the experiment, and logs
of the results, such as the evaluation metrics or TensorBoard logs.

All experiments are stored in the `log.root_experiments_dir` specified in the config.
Each experiment has a name, which is by default the current datetime with the name
of the config.
However, a custom name can be specified by specifying ``--experiment_name <name>`` at the CLI.
The artifacts related to an experiment is stored in a directory with the experiment name
in the `root_experiments_dir`.
"""

from pathlib import Path
from typing import Callable, List, Optional, Tuple, Type
import yaml

import pandas as pd

from quant import Hook, MetricDict
from quant.utils.utils import noop
from quant.data.data_loaders import QuantDataLoader


def log_metrics_to_experiments_dir(
    train_epoch_metrics: List[dict],
    test_epoch_metrics: List[dict],
    experiment_root_directory: Path,
    experiment_name: str,
    skip_training: bool = False,
) -> None:
    """
    Log metrics to experiments directory.

    Args:
        train_epoch_metrics: List of training metrics for every epoch
        test_epoch_metrics: List of test metrics for every epoch
        experiment_root_directory: root directory for storing logs, checkpoints, etc.
        experiment_name: Name of experiment
        skip_training: whether to log only eval metrics
    """
    metrics_dir = experiment_root_directory / experiment_name / 'metrics'
    metrics_dir.mkdir(exist_ok=True, parents=True)

    if not skip_training:
        train_metrics_df = pd.DataFrame.from_records(train_epoch_metrics)
        train_metrics_df.to_csv(metrics_dir / 'train.csv', index=False)

    test_metrics_df = pd.DataFrame.from_records(test_epoch_metrics)
    test_metrics_df.to_csv(metrics_dir / 'test.csv', index=False)


class Experiment:
    """A class representing an experiment."""

    def __init__(
        self,
        task_fn: Callable,
        config: dict,
        data_loader_cls: Type[QuantDataLoader],
        get_hooks: Callable[[dict, Path, MetricDict], Tuple[List[Hook], List[Hook]]],
    ):
        """
        Create an experiment.

        Args:
            task_fn: A function that runs a task, such as classification_task
            config: merged config with CLI args
            data_loader_cls: The QuantDataLoader class
            get_hooks: A function that returns a list of training and testing hooks
        """
        self.task_fn = task_fn
        self.config = config
        self.data_loader_cls = data_loader_cls
        self.get_hooks = get_hooks
        self.name = config['experiment_name']

    def run(
        self,
        logging_root_dir: Path,
        restore_experiment_setup: Callable[[dict], Optional[Path]] = noop,
        restore_experiment_cleanup: Callable[[dict], None] = noop,
    ) -> None:
        """
        Run the experiment.

        Args:
            logging_root_dir: the root logging directory
            restore_experiment_setup: A function that sets
                up the experiment directory to restore, defaults to no-op
            restore_experiment_cleanup: A function that cleans up
                the experiment directory to restore, defaults to no-op
        """
        experiments_dir = logging_root_dir / self.config['experiment_name']
        experiments_dir.mkdir(exist_ok=True, parents=True)
        with open(experiments_dir / 'config.yaml', 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)

        restored_experiment_path = restore_experiment_setup(self.config)

        train_epoch_metrics, test_epoch_metrics = self.task_fn(
            self.config,
            logging_root_dir,
            self.data_loader_cls,
            self.get_hooks,
            restored_experiment_path,
        )

        # Write metrics to experiments directory
        log_metrics_to_experiments_dir(
            train_epoch_metrics,
            test_epoch_metrics,
            logging_root_dir,
            self.name,
            self.config['skip_training']
        )

        restore_experiment_cleanup(self.config)
