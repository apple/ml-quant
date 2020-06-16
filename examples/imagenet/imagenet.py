#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

"""Driver script for running ImageNet."""

from quant.common.compute_platform import LocalComputePlatform
from quant.common.experiment import Experiment
from quant.common.parser import get_base_argument_parser, parse_config
from quant.common.tasks import classification_task
from quant.data.data_loaders import ImageNetDataLoader
from quant.utils.visualization import get_tensorboard_hooks


if __name__ == '__main__':
    parser = get_base_argument_parser('Driver script for running ImageNet.')
    args = parser.parse_args()
    config = parse_config(args)
    platform = LocalComputePlatform(config['log'].get('root_experiments_dir', '.'))
    experiment = Experiment(
        classification_task, config, ImageNetDataLoader, get_tensorboard_hooks
    )
    platform.run(experiment)
