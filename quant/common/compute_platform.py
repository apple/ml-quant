#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

"""
A compute platform is an abstraction of a platform on which to run an experiment.

The most common compute platform is just running something locally,
using :class:`LocalComputePlatform`.
However, :class:`ComputePlatform` can be subclassed to run experiments on
other platforms, such as GPU nodes on some cloud service.

After instantiating a platform and an experiment, we just simply call
:meth:`ComputePlatform.run` to run the experiment on the platform.

Driver scripts support the ``--restore-experiment <path-to-experiment>`` option
to restore the latest checkpoint from a previous experiment.
"""

from abc import ABC, abstractmethod
import os
from pathlib import Path
import subprocess
from typing import Callable, Optional

from quant.common.experiment import Experiment
from quant.utils.utils import noop


def setup_restore_experiment(config: dict) -> Optional[Path]:
    """Set the experiment path to restore experiment."""
    if 'restore_experiment' in config:
        return Path(config['restore_experiment'])
    return None


class ComputePlatform(ABC):
    """Abstract class representing the compute platform to launch jobs from."""

    def __init__(self, root_experiments_dir: str):
        """
        Create a compute platform object.

        Args:
            root_experiments_dir: root directory where experiments will be stored
        """
        self.root_experiments_dir = Path(root_experiments_dir)

    @abstractmethod
    def run(self, experiment: Experiment) -> None:
        """
        Run an experiment on the compute platform.

        Args:
            experiment: the experiment to run
        """
        raise NotImplementedError


class LocalComputePlatform(ComputePlatform):
    """Compute platform for running jobs on local machine."""

    def __init__(self, root_experiments_dir: str):
        """
        Create a compute local compute platform object.

        Args:
            root_experiments_dir: root directory where experiments will be stored
        """
        super(LocalComputePlatform, self).__init__(root_experiments_dir)

    def run(
        self,
        experiment: Experiment,
        restore_experiment_setup: Callable[[dict], Optional[Path]] = setup_restore_experiment,
        restore_experiment_cleanup: Callable[[dict], None] = noop,
    ) -> None:
        """
        Run an experiment function on local machine.

        Args:
            experiment: the experiment to run
            restore_experiment_setup: A function that sets
                up the experiment directory to restore, defaults to no-op
            restore_experiment_cleanup: A function that cleans up
                the experiment directory to restore, defaults to no-op
        """
        # Run TensorBoard process in background
        if experiment.config['log'].get('tensorboard'):
            tensorboard_port = os.environ.get('TENSORBOARD_PORT', '6006')
            tensorboard_proc = subprocess.Popen(
                [
                    'tensorboard',
                    '--logdir',
                    experiment.config['log']['tensorboard_root'],
                    '--port',
                    str(tensorboard_port),
                    '--bind_all',
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=os.environ.copy(),
            )

        # Actually launch experiment
        experiment.run(
            self.root_experiments_dir,
            restore_experiment_setup,
            restore_experiment_cleanup,
        )

        if experiment.config['log'].get('tensorboard'):
            tensorboard_proc.terminate()
