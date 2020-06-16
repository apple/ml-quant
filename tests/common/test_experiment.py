#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

"""Test running experiment on platform."""

import pytest

from quant.common.compute_platform import LocalComputePlatform
from quant.common.experiment import Experiment
from quant.common.tasks import classification_task
from quant.utils.visualization import get_tensorboard_hooks

from tests.data.helpers import get_base_config_template, RandomQuantDataLoader


@pytest.mark.slow
def test_run_experiment_on_platform(tmp_path):
    config = get_base_config_template(
        tmp_path, 'dummy_experiment',
        {'x_quant': 'ls-2', 'w_quant': 'ls-1'}
    )

    platform = LocalComputePlatform(str(tmp_path))

    experiment = Experiment(
        classification_task, config, RandomQuantDataLoader, get_tensorboard_hooks
    )
    platform.run(experiment)

    assert (tmp_path / experiment.name / 'config.yaml').exists()
    assert (tmp_path / experiment.name / 'metrics' / 'test.csv').exists()
